import sys
_module = sys.modules[__name__]
del sys
config = _module
main = _module
methods = _module
bic = _module
finetune = _module
gdumb = _module
icarl = _module
joint = _module
rainbow_memory = _module
regularization = _module
models = _module
cifar = _module
imagenet = _module
layers = _module
mnist = _module
utils = _module
augment = _module
data_loader = _module
method_manager = _module
train_utils = _module

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


import logging.config


import random


from collections import defaultdict


import numpy as np


import torch


from torch import nn


from torch.utils.tensorboard import SummaryWriter


from torchvision import transforms


import logging


from copy import deepcopy


import pandas as pd


from torch.utils.data import DataLoader


import torch.nn as nn


import copy


from typing import List


from torch.utils.data import Dataset


from torch import optim


class BiasCorrectionLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.linear.weight.data.fill_(1.0)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        correction = self.linear(x.unsqueeze(dim=2))
        correction = correction.squeeze(dim=2)
        return correction


class ICaRLNet(nn.Module):

    def __init__(self, model, feature_size, n_class):
        super().__init__()
        self.model = model
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_class, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, opt, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        layer = [conv]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + '2d')(num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + '2d')(num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [conv, bn]
        if opt.activetype is not 'None':
            active = getattr(nn, opt.activetype)()
            layer.append(active)
        if opt.bn and opt.preact:
            layer.append(conv)
        self.block = nn.Sequential(*layer)

    def forward(self, input):
        return self.block.forward(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, opt, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1block = ConvBlock(opt=opt, in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2block = ConvBlock(opt=opt, in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1block(x)
        out = self.conv2block(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        expansion = 4
        self.conv1 = ConvBlock(opt=opt, in_channels=inChannels, out_channels=outChannels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = ConvBlock(opt=opt, in_channels=outChannels, out_channels=outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = ConvBlock(opt=opt, in_channels=outChannels, out_channels=outChannels * expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.downsample = downsample

    def forward(self, x):
        _out = self.conv1(x)
        _out = self.conv2(_out)
        _out = self.conv3(_out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        _out = _out + shortcut
        return _out


class ResidualBlock(nn.Module):

    def __init__(self, opt, block, inChannels, outChannels, depth, stride=1):
        super(ResidualBlock, self).__init__()
        if stride != 1 or inChannels != outChannels * block.expansion:
            downsample = ConvBlock(opt=opt, in_channels=inChannels, out_channels=outChannels * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            downsample = None
        self.blocks = nn.Sequential()
        self.blocks.add_module('block0', block(opt, inChannels, outChannels, stride, downsample))
        inChannels = outChannels * block.expansion
        for i in range(1, depth):
            self.blocks.add_module('block{}'.format(i), block(opt, inChannels, outChannels))

    def forward(self, x):
        return self.blocks(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, opt, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1block = ConvBlock(opt=opt, in_channels=inplanes, out_channels=width, kernel_size=1)
        self.conv2block = ConvBlock(opt=opt, in_channels=width, out_channels=width, kernel_size=3, stride=stride, groups=groups, padding=1)
        self.conv3block = ConvBlock(opt=opt, in_channels=width, out_channels=planes * self.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1block(x)
        out = self.conv2block(out)
        out = self.conv3block(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class FCBlock(nn.Module):

    def __init__(self, opt, in_channels, out_channels, bias=False):
        super(FCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_channels
        self.out_features = out_channels
        lin = nn.Linear(in_channels, out_channels, bias=bias)
        layer = [lin]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + '1d')(num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + '1d')(num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps)
                layer = [lin, bn]
        if opt.activetype is not 'None':
            active = getattr(nn, opt.activetype)()
            layer.append(active)
        if opt.bn and opt.preact:
            layer.append(lin)
        self.block = nn.Sequential(*layer)

    def forward(self, input):
        return self.block.forward(input)


def FinalBlock(opt, in_channels, bias=False):
    out_channels = opt.num_classes
    opt = copy.deepcopy(opt)
    if not opt.preact:
        opt.activetype = 'None'
    return FCBlock(opt=opt, in_channels=in_channels, out_channels=out_channels, bias=bias)


def InitialBlock(opt, out_channels, kernel_size, stride=1, padding=0, bias=False):
    in_channels = opt.in_channels
    opt = copy.deepcopy(opt)
    return ConvBlock(opt=opt, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class ResNetBase(nn.Module):

    def __init__(self, opt, block, layers, zero_init_residual=False, groups=1, width_per_group=64):
        super(ResNetBase, self).__init__()
        self.inplanes = 64
        self.opt = opt
        self.groups = groups
        self.base_width = width_per_group
        self.conv1block = InitialBlock(opt, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(opt=opt, block=block, planes=64, blocks=layers[0])
        self.layer2 = self._make_layer(opt=opt, block=block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(opt=opt, block=block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(opt=opt, block=block, planes=512, blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dim_out = in_channels = 512 * block.expansion
        self.fc = FinalBlock(opt=opt, in_channels=512 * block.expansion)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, opt, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(opt=opt, in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride)
        layers = []
        layers.append(block(opt=opt, inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(opt=opt, inplanes=self.inplanes, planes=planes, groups=self.groups, base_width=self.base_width))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1block(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def ResNet(opt):
    if opt.depth == 18:
        model = ResNetBase(opt, BasicBlock, [2, 2, 2, 2])
    elif opt.depth == 34:
        model = ResNetBase(opt, BasicBlock, [3, 4, 6, 3])
    elif opt.depth == 50 and opt.model == 'ResNet':
        model = ResNetBase(opt, Bottleneck, [3, 4, 6, 3])
    elif opt.depth == 101 and opt.model == 'ResNet':
        model = ResNetBase(opt, Bottleneck, [3, 4, 23, 3])
    elif opt.depth == 152:
        model = ResNetBase(opt, Bottleneck, [3, 8, 36, 3])
    elif opt.depth == 50 and opt.model == 'ResNext':
        model = ResNetBase(opt, Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
    elif opt.depth == 101 and opt.model == 'ResNext':
        model = ResNetBase(opt, Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)
    elif opt.depth == 50 and opt.model == 'WideResNet':
        model = ResNetBase(opt, Bottleneck, [3, 4, 6, 3], width_per_group=128)
    elif opt.depth == 101 and opt.model == 'WideResNet':
        model = ResNetBase(opt, Bottleneck, [3, 4, 23, 3], width_per_group=128)
    else:
        assert opt.depth in ['18', '34', '50', '101', '152'] and opt.model in ['ResNet', 'ResNext', 'WideResNet']
    return model


class MLP(nn.Module):

    def __init__(self, opt):
        super(MLP, self).__init__()
        self.input = FCBlock(opt=opt, in_channels=28 * 28 * 3, out_channels=opt.width)
        self.hidden1 = FCBlock(opt=opt, in_channels=opt.width, out_channels=opt.width)
        self.dim_out = opt.width
        self.fc = FinalBlock(opt=opt, in_channels=opt.width)

    def forward(self, _x):
        _out = _x.view(_x.size(0), -1)
        _out = self.input(_out)
        _out = self.hidden1(_out)
        _out = self.fc(_out)
        return _out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BiasCorrectionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ICaRLNet,
     lambda: ([], {'model': _mock_layer(), 'feature_size': 4, 'n_class': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_clovaai_rainbow_memory(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

