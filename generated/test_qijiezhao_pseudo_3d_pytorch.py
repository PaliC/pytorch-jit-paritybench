import sys
_module = sys.modules[__name__]
del sys
p3d_model = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


from torch.autograd import Variable


import math


from functools import partial


def conv_S(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=1,
        padding=padding, bias=False)


def conv_T(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=1,
        padding=padding, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_s=0,
        depth_3d=47, ST_struc=('A', 'B', 'C')):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.depth_3d = depth_3d
        self.ST_struc = ST_struc
        self.len_ST = len(self.ST_struc)
        stride_p = stride
        if not self.downsample == None:
            stride_p = 1, 2, 2
        if n_s < self.depth_3d:
            if n_s == 0:
                stride_p = 1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=
                False, stride=stride_p)
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            if n_s == self.depth_3d:
                stride_p = 2
            else:
                stride_p = 1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=
                False, stride=stride_p)
            self.bn1 = nn.BatchNorm2d(planes)
        self.id = n_s
        self.ST = list(self.ST_struc)[self.id % self.len_ST]
        if self.id < self.depth_3d:
            self.conv2 = conv_S(planes, planes, stride=1, padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(planes)
            self.conv3 = conv_T(planes, planes, stride=1, padding=(1, 0, 0))
            self.bn3 = nn.BatchNorm3d(planes)
        else:
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3,
                stride=1, padding=1, bias=False)
            self.bn_normal = nn.BatchNorm2d(planes)
        if n_s < self.depth_3d:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=
                False)
            self.bn4 = nn.BatchNorm3d(planes * 4)
        else:
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=
                False)
            self.bn4 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

    def ST_B(self, x):
        tmp_x = self.conv2(x)
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x + tmp_x

    def ST_C(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        tmp_x = self.conv3(x)
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)
        return x + tmp_x

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.id < self.depth_3d:
            if self.ST == 'A':
                out = self.ST_A(out)
            elif self.ST == 'B':
                out = self.ST_B(out)
            elif self.ST == 'C':
                out = self.ST_C(out)
        else:
            out = self.conv_normal(out)
            out = self.bn_normal(out)
            out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2),
        out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class P3D(nn.Module):

    def __init__(self, block, layers, modality='RGB', shortcut_type='B',
        num_classes=400, dropout=0.5, ST_struc=('A', 'B', 'C')):
        self.inplanes = 64
        super(P3D, self).__init__()
        self.input_channel = 3 if modality == 'RGB' else 2
        self.ST_struc = ST_struc
        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(
            1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.depth_3d = sum(layers[:3])
        self.bn1 = nn.BatchNorm3d(64)
        self.cnt = 0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2,
            padding=(0, 1, 1))
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), padding=0,
            stride=(2, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type,
            stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type,
            stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type,
            stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.input_size = self.input_channel, 16, 160, 160
        self.input_mean = [0.485, 0.456, 0.406] if modality == 'RGB' else [0.5]
        self.input_std = [0.229, 0.224, 0.225] if modality == 'RGB' else [np
            .mean([0.229, 0.224, 0.225])]

    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160

    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p = stride
        if self.cnt < self.depth_3d:
            if self.cnt == 0:
                stride_p = 1
            else:
                stride_p = 1, 2, 2
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block, planes=
                        planes * block.expansion, stride=stride)
                else:
                    downsample = nn.Sequential(nn.Conv3d(self.inplanes, 
                        planes * block.expansion, kernel_size=1, stride=
                        stride_p, bias=False), nn.BatchNorm3d(planes *
                        block.expansion))
        elif stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes *
                    block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                    block.expansion, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_s=
            self.cnt, depth_3d=self.depth_3d, ST_struc=self.ST_struc))
        self.cnt += 1
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_s=self.cnt,
                depth_3d=self.depth_3d, ST_struc=self.ST_struc))
            self.cnt += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.maxpool_2(self.layer1(x))
        x = self.maxpool_2(self.layer2(x))
        x = self.maxpool_2(self.layer3(x))
        sizes = x.size()
        x = x.view(-1, sizes[1], sizes[3], sizes[4])
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(self.dropout(x))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_qijiezhao_pseudo_3d_pytorch(_paritybench_base):
    pass
