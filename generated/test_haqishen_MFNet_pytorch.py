import sys
_module = sys.modules[__name__]
del sys
MFNet = _module
SegNet = _module
model = _module
run_demo = _module
test = _module
train = _module
MF_dataset = _module
util = _module
augmentation = _module
util = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


import time


import numpy as np


from torch.autograd import Variable


from torch.utils.data import DataLoader


from torch.utils.data.dataset import Dataset


class ConvBnLeakyRelu2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class MiniInception(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left = ConvBnLeakyRelu2d(in_channels, out_channels // 2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels, out_channels // 2, padding=2, dilation=2)
        self.conv2_left = ConvBnLeakyRelu2d(out_channels, out_channels // 2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels, out_channels // 2, padding=2, dilation=2)
        self.conv3_left = ConvBnLeakyRelu2d(out_channels, out_channels // 2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels, out_channels // 2, padding=2, dilation=2)

    def forward(self, x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x


class MFNet(nn.Module):

    def __init__(self, n_class):
        super(MFNet, self).__init__()
        rgb_ch = [16, 48, 48, 96, 96]
        inf_ch = [16, 16, 16, 36, 36]
        self.conv1_rgb = ConvBnLeakyRelu2d(3, rgb_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb = MiniInception(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb = MiniInception(rgb_ch[3], rgb_ch[4])
        self.conv1_inf = ConvBnLeakyRelu2d(1, inf_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])
        self.conv4_inf = MiniInception(inf_ch[2], inf_ch[3])
        self.conv5_inf = MiniInception(inf_ch[3], inf_ch[4])
        self.decode4 = ConvBnLeakyRelu2d(rgb_ch[3] + inf_ch[3], rgb_ch[2] + inf_ch[2])
        self.decode3 = ConvBnLeakyRelu2d(rgb_ch[2] + inf_ch[2], rgb_ch[1] + inf_ch[1])
        self.decode2 = ConvBnLeakyRelu2d(rgb_ch[1] + inf_ch[1], rgb_ch[0] + inf_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(rgb_ch[0] + inf_ch[0], n_class)

    def forward(self, x):
        x_rgb = x[:, :3]
        x_inf = x[:, 3:]
        x_rgb = self.conv1_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb, kernel_size=2, stride=2)
        x_rgb = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb_p2, kernel_size=2, stride=2)
        x_rgb = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb_p3, kernel_size=2, stride=2)
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb_p4, kernel_size=2, stride=2)
        x_rgb = self.conv5_rgb(x_rgb)
        x_inf = self.conv1_inf(x_inf)
        x_inf = F.max_pool2d(x_inf, kernel_size=2, stride=2)
        x_inf = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf = F.max_pool2d(x_inf_p2, kernel_size=2, stride=2)
        x_inf = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        x_inf = F.max_pool2d(x_inf_p3, kernel_size=2, stride=2)
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf = F.max_pool2d(x_inf_p4, kernel_size=2, stride=2)
        x_inf = self.conv5_inf(x_inf)
        x = torch.cat((x_rgb, x_inf), dim=1)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.decode4(x + torch.cat((x_rgb_p4, x_inf_p4), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.decode3(x + torch.cat((x_rgb_p3, x_inf_p3), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.decode2(x + torch.cat((x_rgb_p2, x_inf_p2), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.decode1(x)
        return x


class ConvBnRelu2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SegNet(nn.Module):

    def __init__(self, n_class, in_channels=4):
        super(SegNet, self).__init__()
        chs = [32, 64, 64, 128, 128]
        self.down1 = nn.Sequential(ConvBnRelu2d(in_channels, chs[0]), ConvBnRelu2d(chs[0], chs[0]))
        self.down2 = nn.Sequential(ConvBnRelu2d(chs[0], chs[1]), ConvBnRelu2d(chs[1], chs[1]))
        self.down3 = nn.Sequential(ConvBnRelu2d(chs[1], chs[2]), ConvBnRelu2d(chs[2], chs[2]), ConvBnRelu2d(chs[2], chs[2]))
        self.down4 = nn.Sequential(ConvBnRelu2d(chs[2], chs[3]), ConvBnRelu2d(chs[3], chs[3]), ConvBnRelu2d(chs[3], chs[3]))
        self.down5 = nn.Sequential(ConvBnRelu2d(chs[3], chs[4]), ConvBnRelu2d(chs[4], chs[4]), ConvBnRelu2d(chs[4], chs[4]))
        self.up5 = nn.Sequential(ConvBnRelu2d(chs[4], chs[4]), ConvBnRelu2d(chs[4], chs[4]), ConvBnRelu2d(chs[4], chs[3]))
        self.up4 = nn.Sequential(ConvBnRelu2d(chs[3], chs[3]), ConvBnRelu2d(chs[3], chs[3]), ConvBnRelu2d(chs[3], chs[2]))
        self.up3 = nn.Sequential(ConvBnRelu2d(chs[2], chs[2]), ConvBnRelu2d(chs[2], chs[2]), ConvBnRelu2d(chs[2], chs[1]))
        self.up2 = nn.Sequential(ConvBnRelu2d(chs[1], chs[1]), ConvBnRelu2d(chs[1], chs[0]))
        self.up1 = nn.Sequential(ConvBnRelu2d(chs[0], chs[0]), ConvBnRelu2d(chs[0], n_class))

    def forward(self, x):
        x = self.down1(x)
        x, ind1 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.down2(x)
        x, ind2 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.down3(x)
        x, ind3 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.down4(x)
        x, ind4 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.down5(x)
        x, ind5 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = F.max_unpool2d(x, ind5, 2, 2)
        x = self.up5(x)
        x = F.max_unpool2d(x, ind4, 2, 2)
        x = self.up4(x)
        x = F.max_unpool2d(x, ind3, 2, 2)
        x = self.up3(x)
        x = F.max_unpool2d(x, ind2, 2, 2)
        x = self.up2(x)
        x = F.max_unpool2d(x, ind1, 2, 2)
        x = self.up1(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBnLeakyRelu2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBnRelu2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MFNet,
     lambda: ([], {'n_class': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (MiniInception,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SegNet,
     lambda: ([], {'n_class': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_haqishen_MFNet_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

