import sys
_module = sys.modules[__name__]
del sys
config = _module
mnist_eval = _module
temporal_ensembling = _module
utils = _module

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


from torch.nn.utils import weight_norm


import numpy as np


from torch.autograd import Variable


import torch.nn.functional as F


import matplotlib


import matplotlib.gridspec as gsp


import matplotlib.pyplot as plt


import torchvision.datasets as datasets


import torchvision.transforms as tf


class GaussianNoise(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 28, 28), std=0.05):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape))
        self.std = std

    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise


class CNN(nn.Module):

    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32):
        super(CNN, self).__init__()
        self.fm1 = fm1
        self.fm2 = fm2
        self.std = std
        self.gn = GaussianNoise(batch_size, std=self.std)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(self.fm2 * 7 * 7, 10)

    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GaussianNoise,
     lambda: ([], {'batch_size': 4}),
     lambda: ([torch.rand([4, 4, 28, 28])], {}),
     False),
]

class Test_ferretj_temporal_ensembling(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

