import sys
_module = sys.modules[__name__]
del sys
data = _module
data_augment = _module
data_load = _module
data_load_ots = _module
eval = _module
main = _module
SFNet = _module
layers = _module
train = _module
train_ots = _module
utils = _module
valid = _module
data_load = _module
main = _module
SFNet = _module
layers = _module
test = _module
train = _module
valid = _module
data_load = _module
eval = _module
main = _module
SFNet = _module
layers = _module
train = _module
valid = _module
data_load = _module
eval = _module
main = _module
SFNet = _module
layers = _module
train = _module
valid = _module
setup = _module
warmup_scheduler = _module
run = _module
scheduler = _module

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


import numpy as np


from torchvision.transforms import functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import random


import torch.nn.functional as f


from torch.backends import cudnn


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


import time


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.sgd import SGD


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import ReduceLROnPlateau


class BasicConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


train_size = 1, 3, 256, 256


class AvgPool2d(nn.Module):

    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]

    def extra_repr(self) ->str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(self.kernel_size, self.base_size, self.kernel_size, self.fast_imp)

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = self.base_size, self.base_size
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])
        if self.fast_imp:
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = (w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        return out


class Gap(nn.Module):

    def __init__(self, in_channel, mode) ->None:
        super().__init__()
        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif mode[0] == 'test':
            if mode[1] == 'HIDE':
                self.gap = AvgPool2d(base_size=110)
            elif mode[1] == 'GOPRO':
                self.gap = AvgPool2d(base_size=80)
            elif mode[1] == 'RSBlur':
                self.gap = AvgPool2d(base_size=75)

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.0)
        x_d = x_d * self.fscale_d[None, :, None, None]
        return x_d + x_h


class Patch_ap(nn.Module):

    def __init__(self, mode, inchannel, patch_size):
        super(Patch_ap, self).__init__()
        if mode[0] == 'train':
            self.ap = nn.AdaptiveAvgPool2d((1, 1))
        elif mode[0] == 'test':
            if mode[1] == 'HIDE':
                self.ap = AvgPool2d(base_size=110)
            elif mode[1] == 'GOPRO':
                self.ap = AvgPool2d(base_size=80)
            elif mode[1] == 'RSBlur':
                self.gap = AvgPool2d(base_size=75)
        self.patch_size = patch_size
        self.channel = inchannel * patch_size ** 2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):
        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)
        low = self.ap(patch_x)
        high = (patch_x - low) * self.h[None, :, None, None]
        out = high + low * self.l[None, :, None, None]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)
        return out


class SFconv(nn.Module):

    def __init__(self, features, mode, M=2, r=2, L=32) ->None:
        super().__init__()
        d = max(int(features / r), L)
        self.features = features
        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, features, 1, 1, 0))
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)
        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif mode[0] == 'test':
            if mode[1] == 'HIDE':
                self.gap = AvgPool2d(base_size=110)
            elif mode[1] == 'GOPRO':
                self.gap = AvgPool2d(base_size=80)
            elif mode[1] == 'RSBlur':
                self.gap = AvgPool2d(base_size=75)

    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)
        fea_z = self.fc(emerge)
        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)
        attention_vectors = torch.cat([high_att, low_att], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)
        fea_high = high * high_att
        fea_low = low * low_att
        out = self.out(fea_high + fea_low)
        return out


class dynamic_filter(nn.Module):

    def __init__(self, inchannels, mode, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = SFconv(inchannels, mode)

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group, self.kernel_size ** 2, h * w)
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, mode, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter
        self.dyna = dynamic_filter(in_channel // 2, mode) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel // 2, mode, kernel_size=5) if filter else nn.Identity()
        self.localap = Patch_ap(mode, in_channel // 2, patch_size=2)
        self.global_ap = Gap(in_channel // 2, mode)

    def forward(self, x):
        out = self.conv1(x)
        if self.filter:
            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)
        non_local, local = torch.chunk(out, 2, dim=1)
        non_local = self.global_ap(non_local)
        local = self.localap(local)
        out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x


class EBlock(nn.Module):

    def __init__(self, out_channel, num_res, mode):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel, mode) for _ in range(num_res - 1)]
        layers.append(ResBlock(out_channel, out_channel, mode, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):

    def __init__(self, channel, num_res, mode):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel, mode) for _ in range(num_res - 1)]
        layers.append(ResBlock(channel, channel, mode, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):

    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True), BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True), BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True), BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False), nn.InstanceNorm2d(out_plane, affine=True))

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):

    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class SFNet(nn.Module):

    def __init__(self, mode, num_res=16):
        super(SFNet, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([EBlock(base_channel, num_res, mode), EBlock(base_channel * 2, num_res, mode), EBlock(base_channel * 4, num_res, mode)])
        self.feat_extract = nn.ModuleList([BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1), BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2), BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True), BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)])
        self.Decoder = nn.ModuleList([DBlock(base_channel * 4, num_res, mode), DBlock(base_channel * 2, num_res, mode), DBlock(base_channel, num_res, mode)])
        self.Convs = nn.ModuleList([BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1), BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)])
        self.ConvsOut = nn.ModuleList([BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1), BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1)])
        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FAM,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SCM,
     lambda: ([], {'out_plane': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_c_yn_SFNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

