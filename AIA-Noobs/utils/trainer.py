import torch
from tqdm.auto import tqdm
import numpy as np
import os
from utils.logger import Logger, ActionLogger
import wandb
import random


class Trainer():
    def __init__(self, loaders, device, model, loss, opt, loggers, config):
        self.loaders = loaders
        self.device = device
        self.model = model
        self.loss_fn = loss
        self.opt = opt
        self.loggers = loggers
        self.config = config

        self.config_multi_gpu()

    def config_multi_gpu(self):
        # ref: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        gpu_count = torch.cuda.device_count()
        gpu = self.config['gpu']
        print("Have ", gpu_count, "GPUs! ", "use: ", gpu)
        if gpu_count > 1 and gpu > 1:
            print('Using SyncBatchNorm() and DataParallel')
            # ref: https://blog.csdn.net/weixin_35757704/article/details/118699506
            import os
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '5678'

            torch.distributed.init_process_group(backend='nccl',
                                                 init_method=None,
                                                 world_size=1,
                                                 rank=0,
                                                 store=None,
                                                 group_name='',
                                                 pg_options=None)
            print('torch.distributed.is_initialized() ', torch.distributed.is_initialized())
            print('Use Multi GPU training')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.DataParallel(model)

        self.model = self.model.to(self.device)
        return

    def fit(self):
        self.train(self.loss_fn, self.model, self.loaders, self.loggers)
    
    def train(self, loss_fn, model, loaders, loggers):
        train_loader, val_loader = loaders
        train_logger, val_logger = loggers
        optimizer = self.opt
        epochs = self.config['epochs']
        device = self.device
        verbose = self.config['verbose']
        print('Start training')
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_epoch(train_loader, model, loss_fn, optimizer, device, train_logger, verbose)
            val_loss, val_acc = self.val_epoch(val_loader, model, loss_fn, device, val_logger, verbose)

            train_logger.on_epoch_end(epoch)
            val_logger.on_epoch_end(epoch)
            Logger.wand(train_logger, val_logger)

            # Save model
            best_loss = np.min(val_logger.loss)
            if val_loss == best_loss:
                best_path = os.path.join(self.config['model_dir'], wandb.run.name+self.config['best'])
                torch.save(model.state_dict(), best_path)
                print('Save model: ', best_path)

#             if epoch % config['save_iter'] == 0:
#                 last_path = os.path.join(config['model_dir'], wandb.run.name+config['last'])
#                 torch.save(model.state_dict(), 
#                            last_path)

        Logger.plot(train_logger, val_logger)


    def test(self, loader, model, loss_fn, config, test_logger):
        model.load_state_dict(torch.load(os.path.join(config['model_dir'], 
                                                      wandb.run.name+config['best'])))
        _ = self.val_epoch(loader, model, loss_fn, self.device, test_logger)
        test_logger.on_epoch_end(epoch=0)

        log = {
            "test_loss": test_logger.loss[-1],
            "test_mDice": test_logger.mdice_logs[-1],
            "test_mIoU": test_logger.miou_logs[-1],
        }
        for i in range(config['num_cls']):
            log[f'test_IoU {i}'] = test_logger.iou_logs[-1][i]
            log[f'test_Dice {i}'] = test_logger.dice_logs[-1][i]

        wandb.log(log)
        wandb.finish()

    def train_epoch(self, dataloader, model, loss_fn, optimizer, device, logger=None, verbose=True):
        model.train()
        epoch_loss, epoch_acc = 0, []

        for batch_i, (x, y) in enumerate(tqdm(dataloader, leave=False, disable=not verbose)):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            if logger:
                logger.update(pred, y, loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc.append(get_binary_accuracy(y, pred))

        return epoch_loss/len(dataloader), np.mean(epoch_acc)


    def val_epoch(self, dataloader, model, loss_fn, device, logger=None, verbose=True):
        model.eval()
        epoch_loss, epoch_acc = 0, []
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(tqdm(dataloader, leave=False, disable=not verbose)):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                if logger:
                    logger.update(pred, y, loss)
                epoch_loss += loss.item()
                epoch_acc.append(get_binary_accuracy(y, pred))

            return epoch_loss/len(dataloader), np.mean(epoch_acc)


class ActionTrainer(Trainer):
    def __init__(self, loaders, device, model, loss, opt, loggers, config):
        super().__init__(loaders, device, model, loss, opt, loggers, config)
        if self.config['gpu'] > 1:
            raise ValueError('Not support > 1 GPU yet')
        ckpt_dir = config['model_dir']
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            print("os.makedirs: ", ckpt_dir)

    def train(self, loss_fn, model, loaders, loggers):
        train_loader, val_loader = loaders
        train_logger, val_logger = loggers
        optimizer = self.opt
        epochs = self.config['epochs']
        device = self.device
        verbose = self.config['verbose']
        print('Start training')
        for epoch in tqdm(range(epochs)):
            _, _ = self.train_epoch(train_loader, model, loss_fn, optimizer, device, train_logger, verbose)
            val_loss, val_acc = self.val_epoch(val_loader, model, loss_fn, device, val_logger, verbose)

            train_logger.on_epoch_end(epoch)
            val_logger.on_epoch_end(epoch)
            ActionLogger.wand(train_logger, val_logger)

            # Save model
            best_loss = np.min(val_logger.loss)
            if val_loss <= best_loss:
                best_path = os.path.join(self.config['model_dir'], wandb.run.name + self.config['best'])
                torch.save(model.state_dict(), best_path)
                print('Save best model: ', best_path)
            torch.save(model.state_dict(), os.path.join(self.config['model_dir'], wandb.run.name + ".pth"))

    def test(self, loader, model, loss_fn, config, test_logger):
        model.load_state_dict(torch.load(os.path.join(config['model_dir'],
                                                      wandb.run.name + config['best'])))
        _ = self.val_epoch(loader, model, loss_fn, self.device, test_logger)
        test_logger.on_epoch_end(epoch=0)

        log = {
            "test_loss": test_logger.loss[-1],
            "test_acc": test_logger.acc_logs[-1],
        }

        wandb.log(log)
        wandb.finish()

    def train_epoch(self, dataloader, model, loss_fn, optimizer, device, logger=None, verbose=True):
        model.train()
        epoch_loss, epoch_acc = 0, []
        num_batch = 0
        random.shuffle(dataloader)
        for loader in tqdm(dataloader, leave=False):
            temp_state = None
            for batch_i, (x, y) in enumerate(loader):
            # for batch_i, (x, y) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
                x, y = x.to(device), y.to(device)
                # LSTM need previous state for sequence outputs
                pred, new_state = model(x, self.state_to_device(temp_state)) if batch_i != 0 else model(x, temp_state)
                temp_state = self.state2np(new_state)

                loss = loss_fn(pred, y)
                if logger:
                    logger.update(pred, y, loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batch += 1
                epoch_acc.append(get_binary_accuracy(y, pred))

        return epoch_loss / num_batch, np.mean(epoch_acc)

    def val_epoch(self, dataloader, model, loss_fn, device, logger=None, verbose=True):
        model.eval()
        epoch_loss, epoch_acc = 0, []
        num_batch = 0
        with torch.no_grad():
            for loader in dataloader:
                temp_state = None
                for batch_i, (x, y) in enumerate(loader):
                # for batch_i, (x, y) in enumerate(tqdm(loader, leave=False, disable=not verbose)):
                    x, y = x.to(device), y.to(device)
                    # LSTM need previous state for sequence outputs
                    pred, new_state = model(x, self.state_to_device(temp_state)) if batch_i != 0 else model(x, temp_state)
                    temp_state = self.state2np(new_state)

                    loss = loss_fn(pred, y)
                    if logger:
                        logger.update(pred, y, loss)
                    epoch_loss += loss.item()
                    num_batch += 1
                    epoch_acc.append(get_binary_accuracy(y, pred))

            return epoch_loss / len(dataloader), np.mean(epoch_acc)
        
    def state_to_device(self, state):
        (h, c) = state
        h = torch.tensor(h, dtype=torch.float)
        c = torch.tensor(c, dtype=torch.float)
        return (h.to(self.device), c.to(self.device))
    def state2np(self, state):
        (h, c) = state
        return (h.detach().cpu().numpy(), c.detach().cpu().numpy())


# Average binary accuracy for all pixel in a batch
def get_binary_accuracy(y_true, y_prob):
    return (y_true == y_prob.argmax(dim=1)).sum().item() / y_true.nelement()


def state_to_device(state, device):
    (h, c) = state
    h = torch.tensor(h, dtype=torch.float)
    c = torch.tensor(c, dtype=torch.float)
    return (h.to(device), c.to(device))
def state2np(state):
    (h, c) = state
    return (h.detach().cpu().numpy(), c.detach().cpu().numpy())