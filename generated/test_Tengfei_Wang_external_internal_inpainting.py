import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module

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


import torch


import torch.nn as nn


class ResBlock(nn.Sequential):

    def __init__(self, num_channels, kernel_size, norm_layer):
        super(ResBlock, self).__init__()
        layers = []
        layers += [nn.ReflectionPad2d((kernel_size - 1) // 2), nn.Conv2d(num_channels, num_channels, kernel_size, bias=False), norm_layer(num_channels, affine=True), nn.LeakyReLU(0.2, inplace=True), nn.ReflectionPad2d((kernel_size - 1) // 2), nn.Conv2d(num_channels, num_channels, kernel_size, bias=False), norm_layer(num_channels, affine=True)]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x) + x


class ResNet(nn.Module):

    def __init__(self, input_channels, output_channels, num_blocks, num_channels, kernel_size, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        layers = []
        layers += [nn.ReflectionPad2d((kernel_size - 1) // 2), nn.Conv2d(input_channels, num_channels, kernel_size), nn.LeakyReLU(0.2, inplace=True)]
        for i in range(num_blocks):
            layers += [ResBlock(num_channels, kernel_size, norm_layer)]
        layers += [nn.ReflectionPad2d((kernel_size - 1) // 2), nn.Conv2d(num_channels, num_channels, kernel_size), norm_layer(num_channels, affine=True)]
        layers += [nn.ReflectionPad2d((kernel_size - 1) // 2), nn.Conv2d(num_channels, output_channels, kernel_size), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ResNet,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'num_blocks': 4, 'num_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Tengfei_Wang_external_internal_inpainting(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

