import sys
_module = sys.modules[__name__]
del sys
agent = _module
base = _module
inferencer = _module
trainer = _module
dataloader = _module
base = _module
vctk = _module
indexer = _module
inference = _module
make_indexes = _module
model = _module
again = _module
preprocess = _module
preprocessor = _module
train = _module
util = _module
config = _module
dsp = _module
mylogger = _module
mytorch = _module
parser = _module
transform = _module
vocoder = _module

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


import logging


import torch


import numpy as np


from torch.utils.data import DataLoader


from functools import partial


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


import matplotlib.pyplot as plt


import random


class InstanceNorm(nn.Module):

    def __init__(self, eps=1e-05):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, x, mask=None):
        B, C = x.shape[:2]
        mn = x.view(B, C, -1).mean(-1)
        sd = (x.view(B, C, -1).var(-1) + self.eps).sqrt()
        mn = mn.view(B, C, *((len(x.shape) - 2) * [1]))
        sd = sd.view(B, C, *((len(x.shape) - 2) * [1]))
        return mn, sd

    def forward(self, x, return_mean_std=False):
        mean, std = self.calc_mean_std(x)
        x = (x - mean) / std
        if return_mean_std:
            return x, mean, std
        else:
            return x


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, groups=1, bias=True, w_init_gain='linear', padding_mode='zeros', sn=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ConvNorm2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear', padding_mode='zeros'):
        super().__init__()
        if padding is None:
            if type(kernel_size) is tuple:
                padding = []
                for k in kernel_size:
                    assert k % 2 == 1
                    p = int(dilation * (k - 1) / 2)
                    padding.append(p)
                padding = tuple(padding)
            else:
                assert kernel_size % 2 == 1
                padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class EncConvBlock(nn.Module):

    def __init__(self, c_in, c_h, subsample=1):
        super().__init__()
        self.seq = nn.Sequential(ConvNorm(c_in, c_h, kernel_size=3, stride=1), nn.BatchNorm1d(c_h), nn.LeakyReLU(), ConvNorm(c_h, c_in, kernel_size=3, stride=subsample))
        self.subsample = subsample

    def forward(self, x):
        y = self.seq(x)
        if self.subsample > 1:
            x = F.avg_pool1d(x, kernel_size=self.subsample)
        return x + y


class DecConvBlock(nn.Module):

    def __init__(self, c_in, c_h, c_out, upsample=1):
        super().__init__()
        self.dec_block = nn.Sequential(ConvNorm(c_in, c_h, kernel_size=3, stride=1), nn.BatchNorm1d(c_h), nn.LeakyReLU(), ConvNorm(c_h, c_in, kernel_size=3))
        self.gen_block = nn.Sequential(ConvNorm(c_in, c_h, kernel_size=3, stride=1), nn.BatchNorm1d(c_h), nn.LeakyReLU(), ConvNorm(c_h, c_in, kernel_size=3))
        self.upsample = upsample

    def forward(self, x):
        y = self.dec_block(x)
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample)
            y = F.interpolate(y, scale_factor=self.upsample)
        y = y + self.gen_block(y)
        return x + y


class Encoder(nn.Module):

    def __init__(self, c_in, c_out, n_conv_blocks, c_h, subsample):
        super().__init__()
        self.inorm = InstanceNorm()
        self.conv1d_first = ConvNorm(c_in * 1, c_h)
        self.conv1d_blocks = nn.ModuleList([EncConvBlock(c_h, c_h, subsample=sub) for _, sub in zip(range(n_conv_blocks), subsample)])
        self.out_layer = ConvNorm(c_h, c_out)

    def forward(self, x):
        y = x
        y = y.squeeze(1)
        y = self.conv1d_first(y)
        mns = []
        sds = []
        for block in self.conv1d_blocks:
            y = block(y)
            y, mn, sd = self.inorm(y, return_mean_std=True)
            mns.append(mn)
            sds.append(sd)
        y = self.out_layer(y)
        return y, mns, sds


class Decoder(nn.Module):

    def __init__(self, c_in, c_h, c_out, n_conv_blocks, upsample):
        super().__init__()
        self.in_layer = ConvNorm(c_in, c_h, kernel_size=3)
        self.act = nn.LeakyReLU()
        self.conv_blocks = nn.ModuleList([DecConvBlock(c_h, c_h, c_h, upsample=up) for _, up in zip(range(n_conv_blocks), upsample)])
        self.inorm = InstanceNorm()
        self.rnn = nn.GRU(c_h, c_h, 2)
        self.out_layer = nn.Linear(c_h, c_out)

    def forward(self, enc, cond, return_c=False, return_s=False):
        y1, _, _ = enc
        y2, mns, sds = cond
        mn, sd = self.inorm.calc_mean_std(y2)
        c = self.inorm(y1)
        c_affine = c * sd + mn
        y = self.in_layer(c_affine)
        y = self.act(y)
        for i, (block, mn, sd) in enumerate(zip(self.conv_blocks, mns, sds)):
            y = block(y)
            y = self.inorm(y)
            y = y * sd + mn
        y = torch.cat((mn, y), dim=2)
        y = y.transpose(1, 2)
        y, _ = self.rnn(y)
        y = y[:, 1:, :]
        y = self.out_layer(y)
        y = y.transpose(1, 2)
        if return_c:
            return y, c
        elif return_s:
            mn = torch.cat(mns, -2)
            sd = torch.cat(sds, -2)
            s = mn * sd
            return y, s
        else:
            return y


class VariantSigmoid(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        y = 1 / (1 + torch.exp(-self.alpha * x))
        return y


class NoneAct(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Activation(nn.Module):
    dct = {'none': NoneAct, 'sigmoid': VariantSigmoid, 'tanh': nn.Tanh}

    def __init__(self, act, params=None):
        super().__init__()
        self.act = Activation.dct[act](**params)

    def forward(self, x):
        return self.act(x)


class Model(nn.Module):

    def __init__(self, encoder_params, decoder_params, activation_params):
        super().__init__()
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
        self.act = Activation(**activation_params)

    def forward(self, x, x_cond=None):
        len_x = x.size(2)
        if x_cond is None:
            x_cond = torch.cat((x[:, :, len_x // 2:], x[:, :, :len_x // 2]), axis=2)
        x, x_cond = x[:, None, :, :], x_cond[:, None, :, :]
        enc, mns_enc, sds_enc = self.encoder(x)
        cond, mns_cond, sds_cond = self.encoder(x_cond)
        enc = self.act(enc), mns_enc, sds_enc
        cond = self.act(cond), mns_cond, sds_cond
        y = self.decoder(enc, cond)
        return y

    def inference(self, source, target):
        original_source_len = source.size(-1)
        original_target_len = target.size(-1)
        if original_source_len % 8 != 0:
            source = F.pad(source, (0, 8 - original_source_len % 8), mode='reflect')
        if original_target_len % 8 != 0:
            target = F.pad(target, (0, 8 - original_target_len % 8), mode='reflect')
        x, x_cond = source, target
        x = x[:, None, :, :]
        x_cond = x_cond[:, None, :, :]
        enc, mns_enc, sds_enc = self.encoder(x)
        cond, mns_cond, sds_cond = self.encoder(x_cond)
        enc = self.act(enc), mns_enc, sds_enc
        cond = self.act(cond), mns_cond, sds_cond
        y = self.decoder(enc, cond)
        dec = y[:, :, :original_source_len]
        return dec


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ConvNorm2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecConvBlock,
     lambda: ([], {'c_in': 4, 'c_h': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (EncConvBlock,
     lambda: ([], {'c_in': 4, 'c_h': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (InstanceNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoneAct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VariantSigmoid,
     lambda: ([], {'alpha': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_KimythAnly_AGAIN_VC(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

