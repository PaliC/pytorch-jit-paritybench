import sys
_module = sys.modules[__name__]
del sys
bttr = _module
datamodule = _module
datamodule = _module
vocab = _module
lit_bttr = _module
lit_ensemble = _module
model = _module
bttr = _module
decoder = _module
encoder = _module
pos_enc = _module
utils = _module
ensemble_test = _module
setup = _module
test = _module
train = _module

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


from typing import List


from typing import Optional


from typing import Tuple


import torch


from torch import FloatTensor


from torch import LongTensor


from torch.utils.data.dataloader import DataLoader


from torchvision.transforms import transforms


import torch.optim as optim


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.transformer import TransformerDecoder


import math


class _Bottleneck(nn.Module):

    def __init__(self, n_channels: 'int', growth_rate: 'int', use_dropout: 'bool'):
        super(_Bottleneck, self).__init__()
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(interChannels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class _SingleLayer(nn.Module):

    def __init__(self, n_channels: 'int', growth_rate: 'int', use_dropout: 'bool'):
        super(_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class _Transition(nn.Module):

    def __init__(self, n_channels: 'int', n_out_channels: 'int', use_dropout: 'bool'):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):

    def __init__(self, growth_rate: 'int', num_layers: 'int', reduction: 'float'=0.5, bottleneck: 'bool'=True, use_dropout: 'bool'=True):
        super(DenseNet, self).__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(1, n_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)
        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)
        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        out = self.post_norm(out)
        return out, out_mask


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DenseNet,
     lambda: ([], {'growth_rate': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 1, 64, 64]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Bottleneck,
     lambda: ([], {'n_channels': 4, 'growth_rate': 4, 'use_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_SingleLayer,
     lambda: ([], {'n_channels': 4, 'growth_rate': 4, 'use_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Transition,
     lambda: ([], {'n_channels': 4, 'n_out_channels': 4, 'use_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Green_Wood_BTTR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

