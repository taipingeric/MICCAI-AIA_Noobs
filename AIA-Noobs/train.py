import argparse
import pandas as pd
import torch
import os
from pprint import pprint

from utils import load_config, check_save_dir
from utils.logger import init_logger
from utils.trainer import Trainer
from utils.loss import build_loss
from models import build_model
from data import build_seg_datasets, build_loader

from ranger21 import Ranger21

def train(args):
    print('torch ', torch.__version__)  # 1.11.0
    print('cuda ', torch.version.cuda)  # 11.3
    print('cudnn ', torch.backends.cudnn.version())  # 8200

    config = load_config(args)
    config['bs'] = args.bs
    pprint(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    train_ds, val_ds = build_seg_datasets(config)
    print(f'train: val = {len(train_ds)}, {len(val_ds)}')

    # init save dir
    check_save_dir(config)

    loaders = build_loader([train_ds, val_ds], config, args)
    
    num_batches_per_epoch=len(train_ds)//config['bs']
    print(num_batches_per_epoch)
    loss_fn = build_loss(config)
    model = build_model(config)
    
    ## Optimizer settings
    cfg_opt=config[config['optm_type']]
    print(cfg_opt)
    if config['optm_type']=='adam':
        opt = torch.optim.Adam(model.parameters(), **cfg_opt)
    elif config['optm_type']=='sgd':
        opt = torch.optim.SGD(model.parameters(), **cfg_opt)
    elif config['optm_type']=='ranger21':
        opt = Ranger21(model.parameters(),
                       num_epochs=config['epochs'],
                       num_batches_per_epoch=num_batches_per_epoch, **cfg_opt)
    else:
        raise AssertionError("No such optimizer!!!")

    loggers = init_logger(config)

    trainer = Trainer(loaders, device, model, loss_fn, opt, loggers, config)
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml')
    parser.add_argument('--config_workspace', type=str, default='./configs/config_aia.yaml')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--bs', type=int, default=3)
    args = parser.parse_args()
    print('Train \n', args)

    train(args)


