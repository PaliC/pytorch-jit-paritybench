import sys
_module = sys.modules[__name__]
del sys
master = _module
cleaning = _module
unet = _module
Unet = _module
unet_model = _module
unet_parts = _module
models = _module
scripts = _module
fine_tuning = _module
fine_tuning_two_network_added_part = _module
generate_synthetic_data = _module
main_cleaning = _module
run = _module
utils = _module
dataloader = _module
loss = _module
synthetic_data_generation = _module
find_duplicates = _module
make_patches = _module
prepare_test_set = _module
preprocess = _module
remove_duplicates_and_trim = _module
render_step_file = _module
render_utils = _module
topology_utils = _module
prepare_patches = _module
preprocess = _module
rasterbg2vectorbg = _module
precision_floorplan_download = _module
merging_for_curves = _module
merging_for_lines = _module
merging_functions = _module
refinement_for_curves = _module
refinement_for_lines = _module
lines_refinement_functions = _module
run_pipeline = _module
util_files = _module
color_utils = _module
data = _module
chunked = _module
graphics = _module
path = _module
primitives = _module
raster_embedded = _module
units = _module
common = _module
parse = _module
raster_utils = _module
splitting = _module
graphics_primitives = _module
line_drawings_dataset = _module
prefetcher = _module
preprocessed = _module
preprocessing = _module
syndata = _module
datasets = _module
patch_topology = _module
snapping = _module
types = _module
transforms = _module
degradation_models = _module
kanungo_degrade = _module
ocrodeg_degrade = _module
raster_transforms = _module
vectordata = _module
prepatch = _module
iterators = _module
prepatched = _module
processing = _module
prepatching = _module
vectortools = _module
dataloading = _module
evaluation_utils = _module
geometric = _module
job_tuples = _module
calculate_metrics = _module
calculate_results = _module
calculate_results_for_curves = _module
logging = _module
loss_functions = _module
lovacz_losses = _module
supervised = _module
metrics = _module
ChamferDistance = _module
chamferdist = _module
iou = _module
raster_metrics = _module
skeleton_metrics = _module
vector_metrics = _module
optimization = _module
energy = _module
gaussian = _module
optimizer = _module
adam = _module
primitive_aligner = _module
scheduled_optimizer = _module
parameters = _module
line_tensor = _module
procedures = _module
primitive_tensor = _module
quadratic_bezier_tensor = _module
canonicals_with_cardano = _module
canonicals_with_probes = _module
energy_with_polyline = _module
energy_with_quadratures = _module
sync_parameters_dirty = _module
os = _module
patchify = _module
cairo = _module
skeleton = _module
simplification = _module
curve = _module
detect_overlaps = _module
join_qb = _module
polyline = _module
simplify = _module
tensorboard = _module
visualization = _module
warnings = _module
vectorization = _module
common = _module
fully_conv_net = _module
generic = _module
lstm = _module
modules = _module
_transformer_modules = _module
base = _module
conv_modules = _module
fully_connected = _module
maybe_module = _module
output = _module
transformer = _module
train_vectorization = _module

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


import torch.functional as F


from torch.autograd import Variable


import numpy as np


import torch.nn.functional as F


from time import gmtime


from time import strftime


import torchvision


from torch.utils.data import DataLoader


from torchvision import transforms


from itertools import product


import random


from torch.utils.data.dataset import Dataset


from time import time


from abc import ABC


from abc import abstractmethod


from matplotlib import pyplot as plt


import math


from torch.multiprocessing import Process


from typing import Dict


from typing import List


from torch.utils.data import Dataset


from collections import OrderedDict


from collections import defaultdict


from enum import Enum


from enum import auto


from numpy.random import uniform


from numpy.random import normal


from torch.utils.data import ConcatDataset


import torch.nn


from functools import partial


from torch import nn


from torch.autograd import Function


import matplotlib.pyplot as plt


from numbers import Number


from typing import Callable


from typing import Iterable


from typing import Tuple


import torchvision.models as models


import torch.nn.init as init


from itertools import islice


import torch.optim


def CreateConvBnRelu(in_channels, out_channels, dilation=1):
    module = nn.Sequential()
    module.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False, dilation=dilation, padding=dilation))
    module.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    module.add_module('relu', nn.ReLU())
    return module


class SmallUnet(nn.Module):

    def __init__(self):
        super(SmallUnet, self).__init__()
        self.add_module('EncConvBnRelu1_1', CreateConvBnRelu(3, 64))
        self.add_module('EncConvBnRelu1_2', CreateConvBnRelu(64, 64))
        self.add_module('EncMp1', nn.MaxPool2d(kernel_size=2))
        self.add_module('EncConvBnRelu2_1', CreateConvBnRelu(64, 128))
        self.add_module('EncConvBnRelu2_2', CreateConvBnRelu(128, 128))
        self.add_module('EncMp2', nn.MaxPool2d(kernel_size=2))
        self.add_module('EncConvBnRelu3_1', CreateConvBnRelu(128, 256))
        self.add_module('EncConvBnRelu3_2', CreateConvBnRelu(256, 256))
        self.add_module('EncMp3', nn.MaxPool2d(kernel_size=2))
        self.add_module('ConvBnRelu4_1', CreateConvBnRelu(256, 512))
        self.add_module('ConvBnRelu4_2', CreateConvBnRelu(512, 512))
        self.add_module('Us4', nn.Upsample(scale_factor=2))
        self.add_module('DecConvBnRelu3_1', CreateConvBnRelu(512 + 256, 256))
        self.add_module('DecConvBnRelu3_2', CreateConvBnRelu(256, 256))
        self.add_module('DecUs3', nn.Upsample(scale_factor=2))
        self.add_module('DecConvBnRelu2_1', CreateConvBnRelu(256 + 128, 128))
        self.add_module('DecConvBnRelu2_2', CreateConvBnRelu(128, 128))
        self.add_module('DecUs2', nn.Upsample(scale_factor=2))
        self.add_module('PredConvBnRelu_1', CreateConvBnRelu(128 + 64, 64))
        self.add_module('PredConvBnRelu_2', CreateConvBnRelu(64, 64))
        self.add_module('PredDense', nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2))
        self.add_module('PredProbs', nn.Sigmoid())

    def forward(self, x):
        enc_1 = self.EncConvBnRelu1_2(self.EncConvBnRelu1_1(x))
        x = self.EncMp1(enc_1)
        enc_2 = self.EncConvBnRelu2_2(self.EncConvBnRelu2_1(x))
        x = self.EncMp2(enc_2)
        enc_3 = self.EncConvBnRelu3_2(self.EncConvBnRelu3_1(x))
        x = self.EncMp2(enc_3)
        x = self.ConvBnRelu4_2(self.ConvBnRelu4_1(x))
        x = self.Us4(x)
        x = torch.cat((x, enc_3), dim=1)
        x = self.DecConvBnRelu3_2(self.DecConvBnRelu3_1(x))
        x = self.DecUs3(x)
        x = torch.cat((x, enc_2), dim=1)
        x = self.DecConvBnRelu2_2(self.DecConvBnRelu2_1(x))
        x = self.DecUs2(x)
        x = torch.cat((x, enc_1), dim=1)
        x = self.PredConvBnRelu_2(self.PredConvBnRelu_1(x))
        x = self.PredDense(x)
        x = self.PredProbs(x)
        x = x.squeeze(dim=1)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True, final_tanh=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.final_tanh = final_tanh

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.final_tanh:
            logits = (F.tanh(logits) + 1) / 2.0
        return logits


class CleaningLoss(nn.Module):

    def __init__(self, kind='MSE', with_restore=True, alpha=1):
        super().__init__()
        if kind == 'MSE':
            self.loss_extraction = nn.MSELoss()
            self.loss_restoration = nn.MSELoss()
        else:
            self.loss_extraction = nn.BCELoss()
            self.loss_restoration = nn.BCELoss()
        self.alpha = alpha
        self.with_restor = with_restore

    def forward(self, y_pred_extract, y_pred_restore, y_true_extract, y_true_restore):
        loss = 0
        y_true_extract = y_true_extract.unsqueeze(1)
        y_true_restore = y_true_restore.unsqueeze(1)
        if self.with_restor:
            loss += self.loss_restoration(y_pred_restore, y_true_restore)
        loss += self.loss_extraction(y_pred_extract, y_true_extract)
        return loss


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class chamferFunction(Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        dist1 = dist1
        dist2 = dist2
        idx1 = idx1
        idx2 = idx2
        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, idx1_, idx2_):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        gradxyz1 = gradxyz1
        gradxyz2 = gradxyz2
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDistance(nn.Module):

    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, input1, input2):
        return chamferFunction.apply(input1, input2)


class SpecifiedModuleBase(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def from_model_spec(cls, spec):
        return cls(**spec)


class MaybeModule(nn.Module):

    def __init__(self, maybe=False, layer=None):
        super().__init__()
        self.maybe = maybe
        self.layer = layer

    def forward(self, input):
        if self.maybe:
            return self.layer.forward(input)
        return input

    def __repr__(self):
        main_str = '{}({}) {}'.format(self._get_name(), self.maybe, repr(self.layer))
        return main_str


class ResnetBlock(nn.Module):

    def __init__(self, resample=None):
        super(ResnetBlock, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.resample = resample

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.resample is not None:
            identity = self.resample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(ResnetBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, resample=None, bn=True, expand=True):
        super(Bottleneck, self).__init__(resample=resample)
        self.conv = nn.Sequential(models.resnet.conv1x1(inplanes, planes), MaybeModule(bn, nn.BatchNorm2d(planes)), nn.LeakyReLU(inplace=True), models.resnet.conv3x3(planes, planes, stride), MaybeModule(bn, nn.BatchNorm2d(planes)), nn.LeakyReLU(inplace=True), models.resnet.conv1x1(planes, planes * (self.expansion if expand else 1)), MaybeModule(bn, nn.BatchNorm2d(planes * (self.expansion if expand else 1))))


resnet101 = Bottleneck, [64, 128, 256, 512], [3, 4, 23, 3], [1, 2, 2, 2]


resnet152 = Bottleneck, [64, 128, 256, 512], [3, 8, 36, 3], [1, 2, 2, 2]


class BasicBlock(ResnetBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, resample=None, bn=True, expand=True):
        super(BasicBlock, self).__init__(resample=resample)
        self.conv = nn.Sequential(models.resnet.conv3x3(inplanes, planes, stride), MaybeModule(bn, nn.BatchNorm2d(planes)), nn.LeakyReLU(inplace=True), models.resnet.conv3x3(planes, planes), MaybeModule(bn, nn.BatchNorm2d(planes)))


resnet18 = BasicBlock, [64, 128, 256, 512], [2, 2, 2, 2], [1, 2, 2, 2]


resnet34 = BasicBlock, [64, 128, 256, 512], [3, 4, 6, 3], [1, 2, 2, 2]


resnet50 = Bottleneck, [64, 128, 256, 512], [3, 4, 6, 3], [1, 2, 2, 2]


def resnet_model_creator(in_channels=1, convmap_channels=128, blocks=1, conf='18', bn=True):
    block_class, blocks_out_channels, blocks_in_layer, stride_in_layer = {'18': resnet18, '34': resnet34, '50': resnet50, '101': resnet101, '152': resnet152}[conf]
    pre_channels = 64
    pre_layers = nn.Sequential(nn.Conv2d(in_channels, pre_channels, kernel_size=7, stride=2, padding=3, bias=False), MaybeModule(bn, nn.BatchNorm2d(pre_channels)), nn.LeakyReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    resnet_feat = [pre_layers]

    def _make_layer(block, out_channels, blocks, stride=1, in_channels=1, bn=True, convmap_channels=None):
        resample1 = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            resample1 = nn.Sequential(models.resnet.conv1x1(in_channels, out_channels * block.expansion, stride), MaybeModule(bn, nn.BatchNorm2d(out_channels * block.expansion)))
        layers = [block(in_channels, out_channels, stride, resample1, bn=bn)]
        in_channels = out_channels * block_class.expansion
        for _ in range(1, blocks - 1):
            layers.append(block(in_channels, out_channels, bn=bn))
        resampleN = None
        if convmap_channels and convmap_channels != out_channels:
            out_channels = convmap_channels
            resampleN = nn.Sequential(models.resnet.conv1x1(in_channels, out_channels), MaybeModule(bn, nn.BatchNorm2d(out_channels)))
        layers.append(block(in_channels, out_channels, 1, resampleN, bn=bn, expand=False))
        return nn.Sequential(*layers)
    assert 1 <= blocks <= 4, 'number of blocks requested for ResNet model should be 1, 2, 3, or 4'
    in_channels = pre_channels
    for block_idx, out_channels, numblocks, stride in zip(range(0, blocks), blocks_out_channels, blocks_in_layer, stride_in_layer):
        convmap_channels_supplied = convmap_channels if block_idx == blocks - 1 else None
        layer = _make_layer(block_class, out_channels, numblocks, stride, in_channels=in_channels, convmap_channels=convmap_channels_supplied, bn=bn)
        in_channels = out_channels * block_class.expansion
        resnet_feat.append(layer)
    return nn.Sequential(*resnet_feat)


def nth(l, x, n=1):
    """Returns n-th occurrence of x in l"""
    matches = (idx for idx, val in enumerate(l) if val == x)
    return next(islice(matches, n - 1, n), None)


def vgg_model_creator(in_channels=1, convmap_channels=128, blocks=1, conf='A', bn=False):
    cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}[conf]

    def _make_layers(cfg, batch_norm=False, in_channels=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), MaybeModule(batch_norm, nn.BatchNorm2d(v)), nn.LeakyReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    last_pool_idx = nth(cfg, 'M', n=blocks)
    cfg[last_pool_idx - 1] = convmap_channels
    cfg = cfg[:last_pool_idx + 1]
    return _make_layers(cfg, batch_norm=bn, in_channels=in_channels)


model_creator = {'resnet': resnet_model_creator, 'vgg': vgg_model_creator}


class ConvFeatureExtractor(SpecifiedModuleBase):

    def __init__(self, in_channels=1, convmap_channels=128, kind='resnet', blocks=1, conf=None, **kwargs):
        """

        :param in_channels: number of channels in source image (typically 1 or 3)
        :param hidden_dim:
        :param resnet_count:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.conv = model_creator[kind](input_channels=in_channels, convmap_channels=convmap_channels, blocks=blocks, conf=conf, **kwargs)

    def forward(self, images):
        return self.conv(images)


class Hidden(SpecifiedModuleBase):

    def forward(self, conv_features, max_lines):
        pass


class VectorizationOutput(SpecifiedModuleBase):

    def __init__(self, hidden_dim=128, ffn_dim=512, n_head=8, num_layers=10, input_channels=1, output_dim=5, resnet_count=0, **kwargs):
        super().__init__(**kwargs)
        self.final_fc = nn.Linear(hidden_dim, self.output_dim)
        self.final_tanh = nn.Tanh()
        self.final_sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self):
        fc = self.final_fc(self.relu(h_dec))
        coord = (self.final_tanh(fc[:, :, :-1]) + 1.0) / 2.0
        prob = self.final_sigm(fc[:, :, -1]).unsqueeze(-1)
        return torch.cat((coord, prob), dim=-1)


class VectorizationModelBase(SpecifiedModuleBase):

    def __init__(self, features, hidden, output):
        """
        :param input_channels: number of input channels in image
        :param convmap_channels: number of convolutional feature channels extracted from image
        """
        super().__init__()
        self.features = features
        self.hidden = hidden
        self.output = output

    def forward(self, images, n):
        x = self.features(images)
        x = self.hidden(x, n)
        x = self.output(x)
        return x

    @classmethod
    def from_model_spec(cls, spec):
        features = ConvFeatureExtractor.from_model_spec(spec['features'])
        hidden = Hidden
        output = VectorizationOutput.from_model_spec(spec['output'])
        return cls(features, hidden, output)


class GlobalMaxPooling(nn.Module):

    def __init__(self, dim=-1):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)[0]


class GlobalPooling(nn.Module):

    def __init__(self, dim=-1):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, x):
        avg = x.mean(dim=self.dim)
        max = x.max(dim=self.dim)[0]
        min = x.min(dim=self.dim)[0]
        return torch.cat([min, avg, max], dim=-1)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bn=True, conv_class=nn.Conv1d):
        super().__init__()
        bn_class = {nn.Conv1d: nn.BatchNorm1d, nn.Conv2d: nn.BatchNorm2d}[conv_class]
        modules = conv_class(in_channels, out_channels, kernel_size), MaybeModule(bn, bn_class(out_channels)), nn.LeakyReLU(inplace=True)
        self.block = nn.Sequential(*modules)


class ConvAdapter(ConvBlock):

    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__(in_channels, out_channels, kernel_size=(1, 1), bn=bn, conv_class=nn.Conv2d)

    def forward(self, conv_features):
        return self.block(conv_features)


class TransformerAdapter(ConvBlock):

    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__(in_channels, out_channels, kernel_size=1, bn=bn)

    def forward(self, decoded):
        decoded = decoded.transpose(1, 2)
        decoded = self.block(decoded)
        decoded = decoded.transpose(1, 2)
        return decoded


class Attn(nn.Module):

    def __init__(self, method, hidden_size, input_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = torch.ByteTensor(mask).unsqueeze(1)
            attn_energies = attn_energies.masked_fill(mask, -1e+18)
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class ATTNLSTMCell(nn.Module):

    def __init__(self, hidden_size, input_size, encoder_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = Attn('concat', hidden_size, encoder_size)
        self.lstm = nn.LSTMCell(encoder_size + input_size, hidden_size)

    def forward(self, current_input, last_hidden, encoder_outputs):
        last_c, last_h = last_hidden
        attn_weights = self.attn(last_h, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context[:, 0]
        rnn_input = torch.cat((current_input, context), 1)
        hidden = self.lstm(rnn_input, last_hidden)
        return hidden


class ATTNLSTM(nn.Module):

    def __init__(self, hidden_size, input_size, encoder_size):
        super().__init__()
        self.cell = ATTNLSTMCell(hidden_size, input_size, encoder_size)

    def forward(self, input_emb, img_seq, prev_state=None):
        if prev_state is None:
            h0 = torch.zeros(img_seq.size(0), self.cell.hidden_size)
            c0 = torch.zeros(img_seq.size(0), self.cell.hidden_size)
            prev_state = c0, h0
        state_seq = []
        for emb_t in input_emb.transpose(0, 1):
            prev_state = self.cell(emb_t, prev_state, img_seq.transpose(0, 1))
            state_seq.append(prev_state)
        cell_seq, hid_seq = zip(*state_seq)
        return torch.stack(cell_seq, dim=1), torch.stack(hid_seq, dim=1)


class BidirectionalATTNLSTM(nn.Module):

    def __init__(self, hidden_size, input_size, encoder_size):
        super().__init__()
        self.lstm_fw = ATTNLSTM(hidden_size, input_size, encoder_size)
        self.lstm_bw = ATTNLSTM(hidden_size, input_size, encoder_size)

    def forward(self, input_emb, img_seq, prev_state_fw=None, prev_state_bw=None):
        cell_seq_fw, hid_seq_fw = self.lstm_fw(input_emb, img_seq, prev_state_fw)
        rev_indices = torch.arange(input_emb.shape[1] - 1, -1, -1)
        cell_seq_bw, hid_seq_bw = self.lstm_bw(input_emb[:, rev_indices], img_seq, prev_state_bw)
        cell_seq = torch.cat([cell_seq_fw, cell_seq_bw], dim=-1)
        hid_seq = torch.cat([hid_seq_fw, hid_seq_bw], dim=-1)
        return cell_seq, hid_seq


class Linear(nn.Module):
    """ Simple Linear layer with xavier init """

    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class Bottle(nn.Module):
    """ Perform the reshape routine before and after an operation """

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    """ Perform the reshape routine before and after a linear projection """
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    """ Perform the reshape routine before and after a softmax operation"""
    pass


class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=0.001):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out


class BatchBottle(nn.Module):
    """ Perform the reshape routine before and after an operation """

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0] * size[1]))
        return out.view(-1, size[0], size[1])


class BottleLayerNormalization(BatchBottle, LayerNormalization):
    """ Perform the reshape routine before and after a layer normalization"""
    pass


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, d_model, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k=None, d_v=None, d_out=None, use_residual=True, dropout=0.0):
        """
        :param n_head: number of parallel attentions (to be concatenated)
        :param d_model: previous layer's last dimension
        :param d_k: size of query and key vectors, defaults to d_model
        :param d_v: size of value vectors, defaults to d_model
        :param d_out: size of output vector, defaults to d_model
        :param use_residual: if True, adds q to output (only works if shapes of q and v are equal)
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k = d_k or d_model
        self.d_v = d_v = d_v or d_model
        self.d_out = d_out = d_out or d_model
        self.use_residual = use_residual
        if self.use_residual:
            assert d_model == d_out, 'if you want residual, input must be of same size as output'
        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))
        self.attention = ScaledDotProductAttention(d_model)
        self.proj = Linear(n_head * d_v, d_out)
        self.layer_norm = LayerNormalization(d_out)
        self.dropout = nn.Dropout(dropout)
        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k=None, v=None, attn_mask=None, return_attn=False):
        """
        :param q: source of queries (w_qs is applied to this vector), also added to output if residual==True
        :type q: Variable(FloatTensor), shape=[batch, n_inp, hid_size]
        :param k: source of keys (w_ks is applied to this vector), defaults to same as q
        :type k: Variable(FloatTensor), shape=[batch, n_inp, hid_size]
        :param v: source of values (w_vs is applied to this vector), defaults to same as q
        :type v: Variable(FloatTensor), shape=[batch, n_inp, hid_size]
        :param attn_mask: if mask is 0, forbids attention to this ninp
        :type mask: Variable(BoolTensor), shape=[batch, n_inp_q, n_inp_kv]
        """
        k = k if k is not None else q
        v = v if v is not None else q
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        residual = q
        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=None if attn_mask is None else attn_mask.repeat(n_head, 1, 1))
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        if self.use_residual:
            outputs = outputs + residual
        outputs = self.layer_norm(outputs)
        return (outputs, attns) if return_attn else outputs


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_hid, d_inner_hid=None, d_out=None, use_residual=True, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        d_inner_hid = d_inner_hid or d_hid
        d_out = d_out or d_hid
        self.use_residual = use_residual
        if self.use_residual:
            assert d_hid == d_out, 'if you want residual, input must be of same size as output'
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)
        self.layer_norm = LayerNormalization(d_out)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        return self.layer_norm(output)


class TransformerLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner=None, n_head=8, d_k=None, d_v=None, dropout=0.1):
        super(TransformerLayer, self).__init__()
        d_inner = d_inner or 4 * d_model
        d_k = d_k or d_model
        d_v = d_v or d_model
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_output = self.slf_attn(dec_input, attn_mask=self_attn_mask)
        dec_output = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class ParameterizedModule(torch.nn.Module, ABC):

    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def from_spec(cls, spec):
        return cls(**spec)


class Conv(ParameterizedModule):

    def forward(self, images):
        return self.conv(images)


class ResnetConv(Conv):

    def __init__(self, in_channels=1, convmap_channels=128, blocks=1, conf=None, **kwargs):
        super().__init__(**kwargs)
        self.conv = resnet_model_creator(in_channels=in_channels, convmap_channels=convmap_channels, blocks=blocks, conf=conf)


class VggConv(Conv):

    def __init__(self, in_channels=1, convmap_channels=128, blocks=1, conf=None, bn=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = vgg_model_creator(in_channels=in_channels, convmap_channels=convmap_channels, blocks=blocks, conf=conf, bn=bn)


class _BasicLinear(nn.Module):
    """A n-layer-feed-forward-layer module."""

    def __init__(self, in_features, out_features, normalization='batch_norm', dropout=0.0):
        super(_BasicLinear, self).__init__()
        normalization_modules = {'batch_norm': nn.BatchNorm1d, 'instance_norm': nn.InstanceNorm1d, 'none': lambda num_features: MaybeModule(False, None)}
        self.layer = nn.Sequential(*(nn.Linear(in_features, out_features), normalization_modules[normalization](out_features), nn.LeakyReLU(inplace=True), nn.Dropout(dropout)))

    def forward(self, features):
        return self.layer(features)


class LinearBlockSequence(ParameterizedModule):

    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.Sequential(*layers)

    @classmethod
    def from_spec(cls, spec):
        layers = [_BasicLinear(**layer_spec) for layer_spec in spec['layers']]
        return cls(layers)

    def forward(self, conv_features, max_lines):
        """
        :param conv_features: [b, h * w, c] batch of image conv features
        :param max_lines: how many lines per image to predict
        """
        out = conv_features.reshape(conv_features.shape[0], -1)
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(out.shape[0], max_lines, -1)
        return out


class Output(ParameterizedModule):

    def __init__(self, in_features=128, out_features=6, **kwargs):
        super().__init__(**kwargs)
        self.final_fc = nn.Linear(in_features, out_features)
        self.final_tanh = nn.Tanh()
        self.final_sigm = nn.Sigmoid()

    def forward(self, vector_features):
        fc = self.final_fc(vector_features)
        if fc.shape[2] <= 6:
            coord = (self.final_tanh(fc[..., :-1]) + 1.0) / 2.0
        else:
            coord = fc[..., :-1]
        prob = self.final_sigm(fc[..., -1]).unsqueeze(-1)
        return torch.cat((coord, prob), dim=-1)


class _InternalSequentialTransformerDecoder(nn.Module):

    def __init__(self, feature_dim=128, ffn_dim=512, n_head=8, num_layers=1, **kwargs):
        """
        :param feature_dim: Wq, Wk, Wv embedding matrixes share this dimension
        :param ffn_dim: size of FC layers in TransformerLayers
        :param n_head: number of heads in TransformerLayers
        :param num_layers: number of TransformerLayers stacked together
        """
        super(_InternalSequentialTransformerDecoder, self).__init__()
        self.transformer = nn.Sequential(*(TransformerLayer(feature_dim, d_inner=ffn_dim, n_head=n_head) for _ in range(num_layers)))
        self.feature_dim = feature_dim

    def forward(self, conv_features, hidden_encoding):
        h_dec = hidden_encoding
        for layer in self.transformer:
            h_dec = layer(h_dec, conv_features)
        return h_dec


class TransformerBase(ParameterizedModule):

    def __init__(self, feature_dim=128, ffn_dim=512, n_head=8, num_layers=1, **kwargs):
        """
        :param feature_dim: Wq, Wk, Wv embedding matrixes share this dimension
        :param ffn_dim: size of FC layers in TransformerLayers
        :param n_head: number of heads in TransformerLayers
        :param num_layers: number of TransformerLayers stacked together
        """
        super(TransformerBase, self).__init__(**kwargs)
        self.decoder = _InternalSequentialTransformerDecoder(feature_dim=feature_dim, ffn_dim=ffn_dim, n_head=n_head, num_layers=num_layers)
        self.feature_dim = feature_dim


def get_sinusoid_encoding_table(n_position, d_hid, scale=2.0, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, float(scale) * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)], dtype='float32')
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0
    return torch.FloatTensor(sinusoid_table)


class TransformerDecoder(TransformerBase):

    def forward(self, conv_features, max_lines):
        """
        :param conv_features: [b, c, h, w] batch of image conv features
        :param max_lines: how many lines per image to predict
        """
        sine_enc = get_sinusoid_encoding_table(max_lines, self.feature_dim, scale=1)[None]
        h_dec = torch.cat([sine_enc] * conv_features.shape[0], dim=0)
        h_dec = h_dec
        decoding = self.decoder(conv_features, h_dec)
        return decoding


class TransformerDiscriminator(TransformerBase):
    LINE_DIM = 6

    def __init__(self, **kwargs):
        super(TransformerDiscriminator, self).__init__(**kwargs)
        self.fc = nn.Linear(self.LINE_DIM, self.feature_dim)

    def forward(self, conv_features, predicted_lines):
        """
        :param conv_features: [b, c, h, w] batch of image conv features
        :param predicted_lines: [b, n, line_dim] batch of predicted n lines per image
        """
        h_dec = self.fc(predicted_lines)
        decoding = self.decoder(conv_features, h_dec)
        return decoding


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ATTNLSTM,
     lambda: ([], {'hidden_size': 4, 'input_size': 4, 'encoder_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Attn,
     lambda: ([], {'method': 4, 'hidden_size': 4, 'input_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BidirectionalATTNLSTM,
     lambda: ([], {'hidden_size': 4, 'input_size': 4, 'encoder_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BottleLayerNormalization,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (BottleLinear,
     lambda: ([], {'d_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (BottleSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CleaningLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvAdapter,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Down,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalMaxPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hidden,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNormalization,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'d_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaybeModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (OutConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ScaledDotProductAttention,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (SmallUnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerAdapter,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (TransformerLayer,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (UNet,
     lambda: ([], {'n_channels': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Up,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (_BasicLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_Vahe1994_Deep_Vectorization_of_Technical_Drawings(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

