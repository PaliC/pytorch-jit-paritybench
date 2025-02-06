import sys
_module = sys.modules[__name__]
del sys
model = _module
run = _module

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


from torch import nn


from torch.autograd import Variable


import torch.nn.functional as F


import time


import torchvision


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.datasets import MNIST


from torchvision.datasets import CIFAR10


from torchvision.utils import save_image


class CDAutoEncoder(nn.Module):
    """
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """

    def __init__(self, input_size, output_size, stride):
        super(CDAutoEncoder, self).__init__()
        self.forward_pass = nn.Sequential(nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0), nn.ReLU())
        self.backward_pass = nn.Sequential(nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=2, padding=0), nn.ReLU())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        x = x.detach()
        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -0.1).type_as(x)
        y = self.forward_pass(x_noisy)
        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    """
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()
        self.ae1 = CDAutoEncoder(3, 128, 2)
        self.ae2 = CDAutoEncoder(128, 256, 2)
        self.ae3 = CDAutoEncoder(256, 512, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)
        if self.training:
            return a3
        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CDAutoEncoder,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (StackedAutoEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_ShayanPersonal_stacked_autoencoder_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

