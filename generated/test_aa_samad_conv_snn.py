import sys
_module = sys.modules[__name__]
del sys
dvscifar_dataloader = _module
main = _module
network = _module
utils = _module
C3D = _module
main = _module
plot_confmat = _module
main_occ = _module
net_model_occ = _module
main_occ = _module
net_model_occ = _module
main = _module
spiking_model = _module
main = _module
net_model = _module
C3D = _module
main = _module
main = _module
net_model = _module
main = _module
net_model = _module
main = _module
net_model = _module
main = _module
spiking_model = _module

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


import numpy as np


import scipy.misc


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from scipy.io import loadmat


import time


from random import randint


import matplotlib.pyplot as plt


import torchvision.transforms as transforms


import torchvision.models as models


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


from torch.optim.lr_scheduler import ReduceLROnPlateau


import math


import torch.utils.model_zoo as model_zoo


import scipy.io


import pandas as pd


import torch.nn.functional as F


import torchvision


from sklearn.metrics import confusion_matrix


import random


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


lens = 0.5 / 3


thresh = 0.5


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = torch.exp(-(input - thresh) ** 2 / (2 * lens ** 2)) / (2 * lens * 3.141592653589793) ** 0.5
        return grad_input * temp.float()


act_fun = ActFun.apply


decay = 0.8


def mem_update(ops, x, mem, spike):
    mem = mem * decay + ops(x)
    spike = act_fun(mem)
    return mem, spike


class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, image_size, batch_size, stride=1, downsample=None):
        super(SpikingBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop2 = nn.Dropout(0.2)
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        w, h = image_size

    def forward(self, x, c1_mem, c1_spike, c2_mem, c2_spike):
        residual = x
        out = self.conv1(x)
        c1_mem, c1_spike = mem_update(out, c1_mem, c1_spike)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        c2_mem, c2_spike = mem_update(out, c2_mem, c2_spike)
        c2_spike = self.drop2(c2_spike)
        return c2_spike, c1_mem, c1_spike, c2_mem, c2_spike


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpikingResNet(nn.Module):

    def __init__(self, block, layers, image_size, batch_size, nb_classes=101, channel=20):
        self.inplanes = 64
        super(SpikingResNet, self).__init__()
        self.nb_classes = nb_classes
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = []
        self.layer_num = layers
        self.size_devide = np.array([4, 4, 4, 4])
        self.planes = [64, 64, 64, 64]
        self._make_layer(block, 64, layers[0], image_size // 4, batch_size)
        self._make_layer(block, 64, layers[1], image_size // 4, batch_size, stride=1)
        self._make_layer(block, 64, layers[2], image_size // 4, batch_size, stride=1)
        self._make_layer(block, 64, layers[3], image_size // 4, batch_size, stride=1)
        self.avgpool2 = nn.AvgPool2d(7)
        self.fc_custom = nn.Linear(64 * 4 * 4 * 4, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(1.0 / n))

    def _make_layer(self, block, planes, blocks, image_size, batch_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
        self.layers.append(block(self.inplanes, planes, image_size, batch_size, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.layers.append(block(self.inplanes, planes, image_size // stride, batch_size))

    def forward(self, input):
        batch_size, time_window, ch, w, h = input.size()
        c_mem = c_spike = torch.zeros(batch_size, 64, w // 2, h // 2, device=device)
        c2_spike, c2_mem, c1_spike, c1_mem = [], [], [], []
        for i in range(len(self.layer_num)):
            d = self.size_devide[i]
            for j in range(self.layer_num[i]):
                c1_mem.append(torch.zeros(batch_size, self.planes[i], w // d, h // d, device=device))
                c1_spike.append(torch.zeros(batch_size, self.planes[i], w // d, h // d, device=device))
                c2_mem.append(torch.zeros(batch_size, self.planes[i], w // d, h // d, device=device))
                c2_spike.append(torch.zeros(batch_size, self.planes[i], w // d, h // d, device=device))
        fc_sumspike = fc_mem = fc_spike = torch.zeros(batch_size, self.nb_classes, device=device)
        for step in range(time_window):
            x = input[:, step, :, :, :]
            x = self.conv1_custom(x)
            c_mem, c_spike = mem_update(x, c_mem, c_spike)
            x = self.avgpool1(c_spike)
            for i in range(len(self.layers)):
                x, c1_mem[i], c1_spike[i], c2_mem[i], c2_spike[i] = self.layers[i](x, c1_mem[i], c1_spike[i], c2_mem[i], c2_spike[i])
            x = torch.cat(c2_spike[0::2], dim=1)
            x = self.avgpool2(x)
            x = x.view(x.size(0), -1)
            out = self.fc_custom(x)
            fc_mem, fc_spike = mem_update(out, fc_mem, fc_spike)
            fc_sumspike += fc_spike
        fc_sumspike = fc_sumspike / time_window
        return fc_sumspike


class C3D(nn.Module):

    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.fc6 = nn.Linear(256 * 2 * 3 * 3, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.__init_weight()

    def forward(self, input, time_window=10):
        x = input[:, :, :, :].float()
        x = x.view(-1, 1, time_window, 28, 28)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = x.view(-1, 256 * 2 * 3 * 3)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        logits = self.fc8(x)
        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


batch_size = 100


cfg_cnn = [(1, 48, 3, 1, 1), (48, 48, 3, 1, 1)]


cfg_fc = [128, 20]


cfg_kernel = [28, 14, 7]


class SCNN(nn.Module):

    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, time_window=10):
        c1_mem = c1_spike = torch.zeros(batch_size * 2, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size * 2, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size * 2, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size * 2, cfg_fc[1], device=device)
        for step in range(time_window):
            x = input[:, step:step + 1, :, :]
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            x = F.avg_pool2d(c1_spike, 2)
            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)
            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size * 2, -1)
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
        outputs = h2_sumspike / time_window
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (C3D,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 1, 10, 28, 28])], {}),
     False),
]

class Test_aa_samad_conv_snn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

