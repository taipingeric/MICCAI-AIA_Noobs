import argparse
import pandas as pd
import torch
import torchvision
import os
from pprint import pprint

from utils import load_config, check_save_dir, load_yaml
from utils.logger import init_action_logger
from utils.trainer import ActionTrainer
from utils.loss import build_loss
# from utils.loss import build_loss
from models.action import CNNLSTM, build_action_encoder
from models import build_model
from data import build_act_datasets, build_action_loader


def train(args):
    print('torch ', torch.__version__)  # 1.11.0
    print('cuda ', torch.version.cuda)  # 11.3
    print('cudnn ', torch.backends.cudnn.version())  # 8200

    config = load_config(args)
    config['bs'] = args.bs
    pprint(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    train_ds, val_ds = build_act_datasets(config)
    print(f'train: val = {len(train_ds)}, {len(val_ds)}')

    # init save dir
    check_save_dir(config)

    loaders = build_action_loader([train_ds, val_ds], config, args)
    loss_fn = torch.nn.CrossEntropyLoss()

    # seg_config = load_yaml(args.config_encoder)
    # seg_model = build_model(seg_config)
    #
    # saved_dict = torch.load(args.ckpt_encoder)
    # # Bug for smp module, key not matched!
    # # remove "module." prefix from saved state dict
    # from collections import OrderedDict
    # updated_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in saved_dict.items()])
    # seg_model.load_state_dict(updated_dict)

    # encoder = seg_model._model.encoder


    # encoder_out = encoder(torch.normal(0, 1, (1, 3, 224, 224)))
    # print(encoder_out)
    # print([i.shape for i in encoder_out])

    # # freeze encoder
    # for param in encoder.parameters():
    #     param.requires_grad = False

    encoder = build_action_encoder(config)
    model = CNNLSTM(config=config, encoder=encoder)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    #
    loggers = init_action_logger(config)
    #
    trainer = ActionTrainer(loaders, device, model, loss_fn, opt, loggers, config)
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml')
    parser.add_argument('--config_workspace', type=str, default='./configs/config_aia.yaml')
    parser.add_argument('--config_encoder', type=str)
    parser.add_argument('--ckpt_encoder', type=str)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--bs', type=int, default=3)  # bs = seq_len
    args = parser.parse_args()
    print('Train \n', args)

    train(args)


