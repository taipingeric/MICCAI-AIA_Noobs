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

    loss_fn = build_loss(config)
    model = build_model(config)

    inputs = torch.normal(0, 1, (1, 3, 480, 480))
    outputs = model(inputs)

    for o in outputs:
        print(o.shape)
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    #
    # loggers = init_logger(config)
    #
    # trainer = Trainer(loaders, device, model, loss_fn, opt, loggers, config)
    # trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml')
    parser.add_argument('--config_workspace', type=str, default='./configs/config_aia.yaml')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--bs', type=int, default=3)
    args = parser.parse_args()
    print('Train \n', args)

    train(args)


