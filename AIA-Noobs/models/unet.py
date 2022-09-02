import torch
import torch.nn as nn
import torch.nn.functional as F
# import timm

from models.cbam import CBAM
from models.elan import ELAN
from models.yolov7 import PoolYolov7, UpYolov7, DecoderYolov7
from models.common import *
from models import head

# from model.transformer import Merger
# from model.unet2plus import UNet2plusEncoder, UNet2plusDecoder




def build_block(config, in_dim, out_dim):
    block = config['block']
    if block == 'basic':
        return ConvBlock(in_dim, out_dim)
    elif block == 'csp':
        return BottleneckCSP(in_dim, out_dim, n=1)
    elif block == 'elan':
        return ELAN(in_dim, out_dim)
    else:
        return


def build_stem(config, in_dim, out_dim):
    """
    first conv block for encoder
    :param config:
    :param in_dim:
    :param out_dim:
    :return:
    """
    block = config['block']
    if block == 'basic':
        return ConvBlock(in_dim, out_dim)
    elif block in ['csp', 'elan']:
        return BottleneckCSP(in_dim, out_dim, n=1)
    else:
        return


def build_norm(config, dim):
    norm = config['model']['norm']
    if norm == 'bn':
        return nn.BatchNorm2d(dim)
    elif norm == 'in':
        return nn.InstanceNorm2d(dim)
    else:
        raise AssertionError("Wrong norm")


class UNet(nn.Module):
    def __init__(self, config, num_cls=10):
        super(UNet, self).__init__()
        print(config)
        self.encoder = Encoder.build(config)  # Encoder(config)
        self.path = PathModule(config)
        self.decoder = Decoder.build(config)
        self.head = Head.build(config, num_cls)

    def forward(self, x):
        features = self.encoder.features(x)
        x = self.path(features)
        x = self.decoder(x)
        x = self.head(x)
        return x


class Head(nn.Module):
    def __init__(self, config, num_cls):
        super().__init__()
        self.head = nn.Conv2d(config['init_dim'],
                              num_cls,
                              config['head']['k_size'])

    def forward(self, x):
        return self.head(x)

    @staticmethod
    def build(config, num_cls):
        head_type = config['head_type']
        print('Head: ', head_type)
        if head_type == 'basic':
            return Head(config, num_cls)
        elif head_type == 'identity':
            return nn.Identity()
        elif head_type== 'cos':
            return head.MetricLayer(config['init_dim'],num_cls)
        else:
            raise AssertionError('Wrong Head typpe')


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = 3
        c = config['init_dim']

        # Encoder
        self.encoder1 = build_stem(config, in_channels, c)  # (3, H, W) -> (c, H, W)

        self.pool1 = build_pooling(config, c)
        self.encoder2 = build_block(config, c, c*2)  # (c, H/2, W/2) -> (2c, H/2, W/2)
        self.pool2 = build_pooling(config, c*2)
        self.encoder3 = build_block(config, c*2, c*4)  # (2c, H/4, W/4) -> (4c, H/4, W/4)
        self.pool3 = build_pooling(config, c*4)
        self.encoder4 = build_block(config, c*4, c*8)  # (4c, H/4, W/4) -> (8c, H/4, W/4)

#         self.pool4 = nn.MaxPool2d(2)
#         self.encoder5 = build_block(config, init_features * 8, init_features * 16)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
#         enc5 = self.encoder5(self.pool4(enc4))

#         return [enc1, enc2, enc3, enc4, enc5]
        return [enc1, enc2, enc3, enc4]
    def features(self, x):
        return self(x)

    @staticmethod
    def build(config):
        encoder = config['encoder_type']
        print('Encoder: ', encoder)
        if encoder == 'basic':
            model = Encoder(config)
        elif encoder == 'cspdarknet53':
            raise AssertionError('Not implemented')
            # model = TIMMEncoder(config)
        elif encoder == 'UNet2plusEncoder':
            raise AssertionError('Not implemented')
            # model = UNet2plusEncoder()
        else:
            raise AssertionError('Wrong Encoder type')
        freeze = config['encoder']['freeze']
        print('freeze: ', freeze)
        if freeze:
            print('Freeze params')
            for param in model.parameters():
                param.requires_grad = False
        return model

def build_pooling(config, c_in):
    pool_type = config['pool_type']
    if pool_type == 'basic':
        return nn.MaxPool2d(2)
    else:
        return PoolYolov7(c_in)

def build_up(config, c_in):
    up_type = config['up_type']
    if up_type == 'yolov7':
        return UpYolov7(c_in)
    else:
        raise AssertionError('Not implemented')



# class TIMMEncoder(nn.Module):
#     """
#     timm model wrapper
#     ref: https://rwightman.github.io/pytorch-image-models/feature_extraction/
#     """
#     def __init__(self, config):
#         super().__init__()
#         arch = config['model']['name']
#         name = config['model'][arch]['encoder']
#         self.stages = config['model'][arch]['stages']
#         assert name != 'basic', 'encoder name'
#         self.encoder = timm.create_model(name,
#                                          features_only=True,
#                                          out_indices=(0, 1, 2, 3, 4),
#                                          pretrained=config['encoder']['pretrained'])

#     def forward(self, x):
#         return self.encoder(x)

#     def features(self, x):
#         return self(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        init_dim = config['init_dim']
        self.stages = config['stages']
        self.up_modules = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(self.stages-1, 0, -1):  # [4, 3, 2, 1]
            self.up_modules.append(nn.ConvTranspose2d(init_dim * (2**i), init_dim * (2**(i-1)),
                                                      kernel_size=2, stride=2))
            self.blocks.append(ConvBlock(init_dim * (2**i), init_dim * (2**(i-1))))

    def forward(self, x):
        assert len(x) == self.stages, f'Decoder input != stages \n len(x) = {len(x)} , self.stages = {self.stages}'

        features = x[::-1]  # enc1~enc4 -> enc4~enc1

        x = features[0]
        for i in range(self.stages-1):
            upper = features[i+1]
            x = torch.cat((self.up_modules[i](x), upper), dim=1)
            x = self.blocks[i](x)
        return x

    @staticmethod
    def build(config):
        decoder_type = config['decoder_type']
        print('Decoder: ', decoder_type)
        if decoder_type == 'basic':
            return Decoder(config)
        elif decoder_type == 'transformer':
            raise AssertionError('Not implemented')
            # return Merger(config)
        elif decoder_type == 'UNet2plusDecoder':
            raise AssertionError('Not implemented')
            # return UNet2plusDecoder(config)
        elif decoder_type == 'yolov7':
            return DecoderYolov7(config)
        else:
            raise AssertionError('Wrong Decoder type')



class PathModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        init_dim = config['init_dim']
        self.stages = config['stages']
        self.cbam = config['path']['cbam']
        if self.cbam:
            print('CBAM')
            self.cbams = nn.ModuleList([CBAM(init_dim*(2**(i))) for i in range(self.stages)])
        else:
            print('Basic Identity path')

    def forward(self, x):
        assert len(x) == self.stages, 'len(x) == self.stages'
        if self.cbam:
            output = []
            for i in range(self.stages):
                att = self.cbams[i]
                output.append(att(x[i]))
            return output
        else:
            return x

if __name__ == '__main__':
    from utils import load_config, load_yaml
    import os

    print('models/init')
    # config = load_config("../")
    print(os.getcwd())

    module = UNet
    inputs = torch.normal(0, 1, (4, 3, 100, 100))
    outputs = module(inputs)
    print(outputs.shape)