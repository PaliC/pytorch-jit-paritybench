import sys
_module = sys.modules[__name__]
del sys
clientALA = _module
serverALA = _module
models = _module
main = _module
ALA = _module
data_utils = _module

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


import copy


import torch


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


from torch.utils.data import DataLoader


from sklearn.preprocessing import label_binarize


from sklearn import metrics


import time


import warnings


import torchvision


import random


from typing import List


from typing import Tuple


class LocalModel(nn.Module):

    def __init__(self, feature_extractor, head):
        super(LocalModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, x, feat=False):
        out = self.feature_extractor(x)
        if feat:
            return out
        else:
            out = self.head(out)
            return out


class FedAvgCNN(nn.Module):

    def __init__(self, in_features=1, num_classes=10, dim=1024, dim1=512):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(2, 2)))
        self.fc1 = nn.Sequential(nn.Linear(dim, dim1), nn.ReLU(inplace=True))
        self.fc = nn.Linear(dim1, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class fastText(nn.Module):

    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        text, text_lengths = x
        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = z
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LocalModel,
     lambda: ([], {'feature_extractor': _mock_layer(), 'head': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_TsingZ0_FedALA(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

