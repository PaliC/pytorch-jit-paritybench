import sys
_module = sys.modules[__name__]
del sys
DiLiGenT_main = _module
PS_Synth_Dataset = _module
datasets = _module
custom_data_loader = _module
pms_transforms = _module
util = _module
run_model = _module
main = _module
PS_FCN = _module
PS_FCN_run = _module
models = _module
custom_model = _module
model_utils = _module
solver_utils = _module
options = _module
base_opts = _module
run_model_opts = _module
train_opts = _module
test_utils = _module
train_utils = _module
utils = _module
eval_utils = _module
logger = _module
recorders = _module
time_utils = _module

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


import scipy.io as sio


import torch


import torch.utils.data as data


import torch.utils.data


import random


import torch.nn as nn


from torch.nn.init import kaiming_normal_


import torchvision.utils as vutils


import math


import time


from collections import OrderedDict


class FeatExtractor(nn.Module):

    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class Regressor(nn.Module):

    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class PS_FCN(nn.Module):

    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(PS_FCN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1:
            light = x[1]
            light_split = torch.split(light, 3, 1)
        feats = torch.Tensor()
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            feat, shape = self.extractor(net_in)
            if i == 0:
                feats = feat
            elif self.fuse_type == 'mean':
                feats = torch.stack([feats, feat], 1).sum(1)
            elif self.fuse_type == 'max':
                feats, _ = torch.stack([feats, feat], 1).max(1)
        if self.fuse_type == 'mean':
            feats = feats / len(img_split)
        feat_fused = feats
        normal = self.regressor(feat_fused, shape)
        return normal

