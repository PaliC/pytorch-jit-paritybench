import sys
_module = sys.modules[__name__]
del sys
lsun = _module
mnist = _module
setup = _module
swae = _module
distributions = _module
models = _module
lsun = _module
mnist = _module
trainer = _module

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


import torch.optim as optim


import torchvision.utils as vutils


from torchvision import datasets


from torchvision import transforms


import matplotlib as mpl


import matplotlib.pyplot as plt


import numpy as np


from sklearn.datasets import make_circles


import torch.nn as nn


import torch.nn.functional as F


class LSUNEncoder(nn.Module):
    """ LSUN Encoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=64, lrelu_slope=0.2, embedding_dim=64):
        super(LSUNEncoder, self).__init__()
        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim
        self.features = nn.Sequential(nn.Conv2d(3, self.init_num_filters_, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_, self.init_num_filters_ * 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.init_num_filters_ * 2), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.init_num_filters_ * 4), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 8, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.init_num_filters_ * 8), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 8, self.embedding_dim_, kernel_size=4, stride=1, padding=0, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.embedding_dim_)
        return x


class LSUNDecoder(nn.Module):
    """ LSUN Decoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=64, embedding_dim=64):
        super(LSUNDecoder, self).__init__()
        self.init_num_filters_ = init_num_filters
        self.embedding_dim_ = embedding_dim
        self.features = nn.Sequential(nn.ConvTranspose2d(self.embedding_dim_, self.init_num_filters_ * 8, kernel_size=4, stride=1, padding=0, bias=False), nn.BatchNorm2d(self.init_num_filters_ * 8), nn.ReLU(inplace=True), nn.ConvTranspose2d(self.init_num_filters_ * 8, self.init_num_filters_ * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.init_num_filters_ * 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(self.init_num_filters_ * 4, self.init_num_filters_ * 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.init_num_filters_ * 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(self.init_num_filters_ * 2, self.init_num_filters_, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.init_num_filters_), nn.ReLU(inplace=True), nn.ConvTranspose2d(self.init_num_filters_, 3, kernel_size=4, stride=2, padding=1, bias=True))

    def forward(self, z):
        z = z.view(-1, self.embedding_dim_, 1, 1)
        z = self.features(z)
        return torch.sigmoid(z)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LSUNAutoencoder(nn.Module):
    """ LSUN Autoencoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=64, lrelu_slope=0.2, embedding_dim=64):
        super(LSUNAutoencoder, self).__init__()
        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.embedding_dim_ = embedding_dim
        self.encoder = LSUNEncoder(init_num_filters, lrelu_slope, embedding_dim)
        self.decoder = LSUNDecoder(init_num_filters, embedding_dim)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class MNISTEncoder(nn.Module):
    """ MNIST Encoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2):
        super(MNISTEncoder, self).__init__()
        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim
        self.features = nn.Sequential(nn.Conv2d(1, self.init_num_filters_ * 1, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.AvgPool2d(kernel_size=2, padding=0), nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 2, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 2, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.AvgPool2d(kernel_size=2, padding=0), nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 4, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.AvgPool2d(kernel_size=2, padding=1))
        self.fc = nn.Sequential(nn.Linear(self.init_num_filters_ * 4 * 4 * 4, self.inter_fc_dim_), nn.ReLU(inplace=True), nn.Linear(self.inter_fc_dim_, self.embedding_dim_))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.init_num_filters_ * 4 * 4 * 4)
        x = self.fc(x)
        return x


class MNISTDecoder(nn.Module):
    """ MNIST Decoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2):
        super(MNISTDecoder, self).__init__()
        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim
        self.fc = nn.Sequential(nn.Linear(self.embedding_dim_, self.inter_fc_dim_), nn.Linear(self.inter_fc_dim_, self.init_num_filters_ * 4 * 4 * 4), nn.ReLU(inplace=True))
        self.features = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Upsample(scale_factor=2), nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=0), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Upsample(scale_factor=2), nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 2, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 2, kernel_size=3, padding=1), nn.LeakyReLU(self.lrelu_slope_, inplace=True), nn.Conv2d(self.init_num_filters_ * 2, 1, kernel_size=3, padding=1))

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 4 * self.init_num_filters_, 4, 4)
        z = self.features(z)
        return F.sigmoid(z)


class MNISTAutoencoder(nn.Module):
    """ MNIST Autoencoder from Original Paper's Keras based Implementation.

        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2):
        super(MNISTAutoencoder, self).__init__()
        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim
        self.encoder = MNISTEncoder(init_num_filters, lrelu_slope, inter_fc_dim, embedding_dim)
        self.decoder = MNISTDecoder(init_num_filters, lrelu_slope, inter_fc_dim, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LSUNAutoencoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (LSUNDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSUNEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (MNISTAutoencoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 24, 24])], {}),
     True),
    (MNISTEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 24, 24])], {}),
     True),
]

class Test_eifuentes_swae_pytorch(_paritybench_base):
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

