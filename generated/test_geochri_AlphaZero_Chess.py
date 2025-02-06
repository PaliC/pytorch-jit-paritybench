import sys
_module = sys.modules[__name__]
del sys
MCTS_chess = _module
alpha_net = _module
analyze_games = _module
chess_board = _module
encoder_decoder = _module
evaluator = _module
pipeline = _module
train = _module
train_multiprocessing = _module
visualize_board = _module

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


import collections


import numpy as np


import math


import copy


import torch


import torch.multiprocessing as mp


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import matplotlib


import matplotlib.pyplot as plt


class ConvBlock(nn.Module):

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8 * 8 * 73
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 22, 8, 8)
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):

    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):

    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8 * 8 * 128, 8 * 8 * 73)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))
        v = v.view(-1, 8 * 8)
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        p = F.relu(self.bn1(self.conv1(s)))
        p = p.view(-1, 8 * 8 * 128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ChessNet(nn.Module):

    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, 'res_%i' % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, 'res_%i' % block)(s)
        s = self.outblock(s)
        return s


class AlphaLoss(torch.nn.Module):

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum(-policy * (1e-06 + y_policy.float()).float().log(), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlphaLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([16, 4]), torch.rand([16, 4]), torch.rand([64, 4]), torch.rand([64, 4])], {}),
     True),
    (ChessNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 22, 8, 8])], {}),
     False),
    (ConvBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 22, 8, 8])], {}),
     True),
    (OutBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (ResBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
]

class Test_geochri_AlphaZero_Chess(_paritybench_base):
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

