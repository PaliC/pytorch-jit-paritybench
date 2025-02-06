import sys
_module = sys.modules[__name__]
del sys
linear = _module
main = _module
main_mixup = _module
model = _module
utils = _module
utils_mixup = _module

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


import torch.optim as optim


from torch.utils.data import DataLoader


from torchvision.datasets import STL10


from torchvision.datasets import CIFAR10


from torchvision.datasets import CIFAR100


import numpy as np


from torch import nn


from torch import optim


from torch import autograd


import torch.nn.functional as F


from torchvision.models.resnet import resnet50


from torchvision import transforms


import math


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim.lr_scheduler import MultiStepLR


from torch.utils import data


import random


class Model(nn.Module):

    def __init__(self, feature_dim=128):
        super(Model, self).__init__()
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Net(nn.Module):

    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()
        model = Model()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(pretrained_path))
        self.f = model.module.f
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_Wangt_CN_IP_IRM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

