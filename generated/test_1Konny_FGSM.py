import sys
_module = sys.modules[__name__]
del sys
adversary = _module
cleaner = _module
datasets = _module
main = _module
toynet = _module
solver = _module
utils = _module
visdom_utils = _module

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


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


from torchvision.utils import save_image


import torch.nn as nn


import numpy as np


from torch import nn


def kaiming_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform(m.weight, a=0, mode='fan_in')
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()


class ToyNet(nn.Module):

    def __init__(self, x_dim=784, y_dim=10):
        super(ToyNet, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.mlp = nn.Sequential(nn.Linear(self.x_dim, 300), nn.ReLU(True),
            nn.Linear(300, 150), nn.ReLU(True), nn.Linear(150, self.y_dim))

    def forward(self, X):
        if X.dim() > 2:
            X = X.view(X.size(0), -1)
        out = self.mlp(X)
        return out

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())


class One_Hot(nn.Module):

    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0, X_in.data))

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.depth)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_1Konny_FGSM(_paritybench_base):
    pass
    def test_000(self):
        self._check(ToyNet(*[], **{}), [torch.rand([784, 784])], {})

