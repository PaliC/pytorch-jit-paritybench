import sys
_module = sys.modules[__name__]
del sys
opt = _module
src = _module
data_loader = _module
eval = _module
gaussian = _module
model = _module
utils = _module
test = _module
train = _module

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


from torch import nn


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class Add(nn.Module):

    def forward(self, input1, input2):
        return torch.add(input1, input2)


class PRN(nn.Module):

    def __init__(self, node_count, coeff):
        super(PRN, self).__init__()
        self.flatten = Flatten()
        self.height = coeff * 28
        self.width = coeff * 18
        self.dens1 = nn.Linear(self.height * self.width * 17, node_count)
        self.bneck = nn.Linear(node_count, node_count)
        self.dens2 = nn.Linear(node_count, self.height * self.width * 17)
        self.drop = nn.Dropout()
        self.add = Add()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        res = self.flatten(x)
        out = self.drop(F.relu(self.dens1(res)))
        out = self.drop(F.relu(self.bneck(out)))
        out = F.relu(self.dens2(out))
        out = self.add(out, res)
        out = self.softmax(out)
        out = out.view(out.size()[0], self.height, self.width, 17)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_salihkaragoz_pose_residual_network_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Add(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

