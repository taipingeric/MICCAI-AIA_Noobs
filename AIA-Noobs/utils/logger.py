import yaml
from ignite.metrics import ConfusionMatrix, DiceCoefficient, IoU, Accuracy
import matplotlib.pyplot as plt
import numpy as np
import wandb
from datetime import datetime
import os


def init_logger(config, init_wandb=True):
    train_logger = Logger(config, 'train', init_wandb)
    val_logger = Logger(config, 'val')
    return train_logger, val_logger


def init_action_logger(config, init_wandb=True):
    train_logger = ActionLogger(config, 'train', init_wandb)
    val_logger = ActionLogger(config, 'val')
    return train_logger, val_logger


class Logger:
    def __init__(self, config, mode='train', init_wandb=True):
        self.mode = mode
        self.config = config
        self.cm = ConfusionMatrix(config['num_cls'])
        self.verbose = config['verbose']
        self.dice = DiceCoefficient(self.cm)
        self.iou = IoU(self.cm)
        self.acc = Accuracy()
        self.batch_loss = []
        self.loss = []
        self.dice_logs = []
        self.mdice_logs = []
        self.iou_logs = []
        self.miou_logs = []
        self.acc_logs = []
        if init_wandb:
            self.init_wandb()
    
    def update(self, y_pred, y_true, loss):
        self.cm.update([y_pred, y_true])
        self.acc.update([y_pred, y_true])
        self.batch_loss.append(loss.cpu().item())
        
    def compute(self):
        dice = self.dice.compute().numpy()
        mdice = dice.mean()
        
        iou = self.iou.compute().numpy()
        miou = iou.mean()
        
        acc = self.acc.compute()
        self.dice_logs.append(list(dice))
        self.mdice_logs.append(mdice)
        self.iou_logs.append(list(iou))
        self.miou_logs.append(miou)
        self.acc_logs.append(acc)
        
        return
    
    def on_epoch_end(self, epoch):
        self.compute()
        
        self.cm.reset()
        self.dice.reset()
        self.loss.append(np.mean(self.batch_loss))
        self.batch_loss = []
        if self.verbose:
            print(f'[EPOCH: {epoch+1:04d}] {self.mode:<7} loss: {self.loss[-1]:.4f}, acc: {self.acc_logs[-1]:.3f}, \
mDice: {self.mdice_logs[-1]:.3f}, mIoU: {self.miou_logs[-1]:.3f}')

    def init_wandb(self):
        if self.mode == 'train':
            wandb.init(project=self.config['project'], 
                       entity=self.config['entity'])
            wandb.config.hparams = self.config
            now = datetime.now()
            date_str = now.strftime("%Y%m%d-%H%M")
            wandb.run.name = date_str
            wandb.run.save()
            self.run_name = date_str
            self.copy_config()

    def copy_config(self):
        config_dir = self.config['config_dir']
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        with open(os.path.join(config_dir, self.run_name+'.yaml'), 'w') as file:
            _ = yaml.dump(self.config, file)


    @staticmethod
    def wand(train_logger, val_logger):
        log = {
            "train_loss": train_logger.loss[-1], "val_loss": val_logger.loss[-1],
            "train_mDice": train_logger.mdice_logs[-1], "val_mDice": val_logger.mdice_logs[-1],
            "train_mIoU": train_logger.miou_logs[-1], "val_mIoU": val_logger.miou_logs[-1],
        }
        for i in range(train_logger.config['num_cls']):
            log[f'val_IoU {i}'] = val_logger.iou_logs[-1][i]
            log[f'val_Dice {i}'] = val_logger.dice_logs[-1][i]
        wandb.log(log)

    @staticmethod
    def plot(train_logger, val_logger):
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.title('Loss')
        plt.plot(train_logger.loss)
        plt.plot(val_logger.loss)
        plt.legend(['train', 'val'])
        plt.subplot(1, 4, 2)
        plt.title('Acc.')
        plt.plot(train_logger.acc_logs)
        plt.plot(val_logger.acc_logs)
        plt.legend(['train', 'val'])
        plt.subplot(1, 4, 3)
        plt.title('mDice')
        plt.plot(train_logger.mdice_logs)
        plt.plot(val_logger.mdice_logs)
        plt.legend(['train', 'val'])
        plt.subplot(1, 4, 4)
        plt.title('mIoU')
        plt.plot(train_logger.miou_logs)
        plt.plot(val_logger.miou_logs)
        plt.legend(['train', 'val'])
        plt.show()

# TODO: F1 score
class ActionLogger:
    def __init__(self, config, mode='train', init_wandb=True):
        self.mode = mode
        self.config = config
        self.cm = ConfusionMatrix(8)
        self.verbose = config['verbose']
        self.acc = Accuracy()
        self.batch_loss = []
        self.loss = []
        self.acc_logs = []
        if init_wandb:
            self.init_wandb()

    def update(self, y_pred, y_true, loss):
        self.cm.update([y_pred, y_true])
        self.acc.update([y_pred, y_true])
        self.batch_loss.append(loss.cpu().item())

    def compute(self):
        acc = self.acc.compute()
        self.acc_logs.append(acc)
        return

    def on_epoch_end(self, epoch):
        self.compute()

        self.cm.reset()
        self.loss.append(np.mean(self.batch_loss))
        self.batch_loss = []
        if self.verbose:
            print(f'[EPOCH: {epoch + 1:04d}] {self.mode:<7} loss: {self.loss[-1]:.4f}, acc: {self.acc_logs[-1]:.3f}, ')

    def init_wandb(self):
        if self.mode == 'train':
            wandb.init(project=self.config['project_action'],
                       entity=self.config['entity'])
            wandb.config.hparams = self.config
            now = datetime.now()
            date_str = now.strftime("%Y%m%d-%H%M")
            wandb.run.name = date_str
            wandb.run.save()
            self.run_name = date_str
            self.copy_config()

    def copy_config(self):
        config_dir = self.config['config_dir']
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        with open(os.path.join(config_dir, self.run_name + '.yaml'), 'w') as file:
            _ = yaml.dump(self.config, file)

    @staticmethod
    def wand(train_logger, val_logger):
        log = {
            "train_loss": train_logger.loss[-1], "val_loss": val_logger.loss[-1],
            "train_acc": train_logger.acc_logs[-1], "val_acc": val_logger.acc_logs[-1],
        }
        wandb.log(log)

    @staticmethod
    def plot(train_logger, val_logger):
        return