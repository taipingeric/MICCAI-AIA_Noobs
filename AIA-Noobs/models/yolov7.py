import torch
import torch.nn as nn
from models.common import Conv
from models.elan import ELAN

'''
pool_type: yolov7
# Down c_out = 256
[-1, 1, MP, []], # MP: MaxPooling
[-1, 1, Conv, [128, 1, 1]],
[-3, 1, Conv, [128, 1, 1]],
[-1, 1, Conv, [128, 3, 2]],
[[-1, -3], 1, Concat, [1]],  # 16-P3/8
c_out = c_in
'''
class PoolYolov7(nn.Module):
    def __init__(self, c_in):
        super(PoolYolov7, self).__init__()
        self.c1 = Conv(c_in, c_in//2, 1)
        self.c_pool = Conv(c_in//2, c_in//2, 3, 2)

        self.pool = nn.MaxPool2d(2)
        self.c2 = Conv(c_in, c_in//2, 1)

    def __call__(self, x):
        x1 = self.pool(x)
        x1 = self.c2(x1)

        x2 = self.c1(x)
        x2 = self.c_pool(x2)

        x = torch.cat([x2, x1], dim=1)
        return x

'''
# *2 Up, 512
[-1, 1, Conv, [256, 1, 1]],
[-1, 1, nn.Upsample, [None, 2, 'nearest']],
[37, 1, Conv, [256, 1, 1]], # route backbone P4
[[-1, -2], 1, Concat, [1]],
c_out = c_in
'''
class UpYolov7(nn.Module):
    def __init__(self, c_in):
        super(UpYolov7, self).__init__()
        c_ = c_in//2
        self.c1 = Conv(c_in, c_, 1)
        self.up = nn.Upsample(None, 2)
        self.c2 = Conv(c_in, c_, 1)

    def __call__(self, x1, x2):
        '''

        :param x1: deeper
        :param x2: shallower
        :return:
        '''
        x1 = self.c1(x1)
        x1 = self.up(x1)

        x2 = self.c2(x2)
        assert x1.shape == x2.shape, 'inconsistece shape for 2 branches fix!'
        return torch.cat([x1, x2], dim=1)


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class DecoderYolov7(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config['init_dim'] # 16
        self.stages = config['stages']
        self.spp = SPPCSPC(c*8, c*4)
        self.up2 = UpYolov7(c*4)
        self.block2 = ELAN(c*4, c*2)

        self.up1 = UpYolov7(c*2)
        self.block1 = ELAN(c*2, c)

        self.up0 = UpYolov7(c)
        self.block0 = ELAN(c, c)
    def __call__(self, x):
        assert len(x) == self.stages, 'inconsistency stages and encoder output'
        enc0, enc1, enc2, enc3 = x  # [c, 2c, 4c, 8c]
        enc3 = self.spp(enc3)  # [4c]
        dec2 = self.up2(enc3, enc2)  # 4c
        dec2 = self.block2(dec2)  # 2c

        dec1 = self.up1(dec2, enc1)
        dec1 = self.block1(dec1)  # c

        dec0 = self.up0(dec1, enc0)
        dec0 = self.block0(dec0)  # c

        return dec0