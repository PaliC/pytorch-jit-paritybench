import sys
_module = sys.modules[__name__]
del sys
loader = _module
main = _module
metric = _module
networks = _module
prep = _module
solver = _module

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


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.backends import cudnn


import torch


from math import exp


import torch.nn.functional as F


from torch.autograd import Variable


import torch.nn as nn


from collections import OrderedDict


from torchvision.models import vgg19


import time


import matplotlib


import matplotlib.pyplot as plt


import torch.optim as optim


class WGAN_VGG_generator(nn.Module):

    def __init__(self):
        super(WGAN_VGG_generator, self).__init__()
        layers = [nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU()]
        for i in range(2, 8):
            layers.append(nn.Conv2d(32, 32, 3, 1, 1))
            layers.append(nn.ReLU())
        layers.extend([nn.Conv2d(32, 1, 3, 1, 1), nn.ReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class WGAN_VGG_discriminator(nn.Module):

    def __init__(self, input_size):
        super(WGAN_VGG_discriminator, self).__init__()

        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n - k) // s + 1
            return n

        def add_block(layers, ch_in, ch_out, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.LeakyReLU())
            return layers
        layers = []
        ch_stride_set = [(1, 64, 1), (64, 64, 2), (64, 128, 1), (128, 128, 2), (128, 256, 1), (256, 256, 2)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)
        self.output_size = conv_output_size(input_size, [3] * 6, [1, 2] * 3)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256 * self.output_size * self.output_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256 * self.output_size * self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out


class WGAN_VGG_FeatureExtractor(nn.Module):

    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature_extractor(x)
        return out


class WGAN_VGG(nn.Module):

    def __init__(self, input_size=64):
        super(WGAN_VGG, self).__init__()
        self.generator = WGAN_VGG_generator()
        self.discriminator = WGAN_VGG_discriminator(input_size)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss()

    def d_loss(self, x, y, gp=True, return_gp=False):
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        fake = self.generator(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + 0.1 * p_loss
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x, y):
        fake = self.generator(x).repeat(1, 3, 1, 1)
        real = y.repeat(1, 3, 1, 1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a * y + (1 - a) * fake).requires_grad_(True)
        d_interp = self.discriminator(interp)
        fake_ = torch.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(outputs=d_interp, inputs=interp, grad_outputs=fake_, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
        return gradient_penalty


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (WGAN_VGG_FeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (WGAN_VGG_discriminator,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 1, 48, 48])], {}),
     True),
    (WGAN_VGG_generator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_SSinyu_WGAN_VGG(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

