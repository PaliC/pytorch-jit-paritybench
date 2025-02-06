import sys
_module = sys.modules[__name__]
del sys
create_mnistm = _module
main = _module
mnist = _module
mnistm = _module
model = _module
params = _module
test = _module
train = _module
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


import torchvision.datasets as datasets


from torch.utils.data import SubsetRandomSampler


from torch.utils.data import DataLoader


from torchvision import transforms


import torch.utils.data as data


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import torch.optim as optim


import matplotlib.pyplot as plt


from torch.autograd import Function


from sklearn.manifold import TSNE


import itertools


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1, 3 * 28 * 28)
        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(in_features=3 * 28 * 28, out_features=100), nn.ReLU(), nn.Linear(in_features=100, out_features=100), nn.ReLU(), nn.Linear(in_features=100, out_features=10))

    def forward(self, x):
        x = self.classifier(x)
        return x


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(nn.Linear(in_features=3 * 28 * 28, out_features=100), nn.ReLU(), nn.Linear(in_features=100, out_features=2))

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return x

