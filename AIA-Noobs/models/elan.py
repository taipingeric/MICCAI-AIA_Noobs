import torch
import torch.nn as nn
from models.common import Conv


"""# ELAN, 256
[-1, 1, Conv, [64, 1, 1]],  # -6
[-2, 1, Conv, [64, 1, 1]],  # -5
[-1, 1, Conv, [64, 3, 1]],
[-1, 1, Conv, [64, 3, 1]],  # -3
[-1, 1, Conv, [64, 3, 1]],
[-1, 1, Conv, [64, 3, 1]],  # -1
[[-1, -3, -5, -6], 1, Concat, [1]],
[-1, 1, Conv, [256, 1, 1]],  # 11
"""
class ELAN(nn.Module):
    """ELAN from yolov7"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        c_ = in_dim // 2
        self.c1 = Conv(in_dim, c_, 1)  # -6
        self.c2 = Conv(in_dim, c_, 1)  # -5
        self.c3 = Conv(c_, c_, 3)
        self.c4 = Conv(c_, c_, 3)  # -3
        self.c5 = Conv(c_, c_, 3)
        self.c6 = Conv(c_, c_, 3)  # -1
        self.cout = Conv(c_*4, out_dim, 3)
    def forward(self, x):
        route1 = self.c1(x)
        x = self.c2(x)
        mid_5 = x
        x = self.c3(x)
        x = self.c4(x)
        mid_3 = x
        x = self.c5(x)
        x = self.c6(x)
        x = torch.cat([route1, mid_5, mid_3, x], dim=1)
        x = self.cout(x)
        return x


if __name__ == '__main__':
    module = ELAN(16, 32)
    inputs = torch.normal(0, 1, (4, 16, 100, 100))
    outputs = module(inputs)
    print(outputs.shape)