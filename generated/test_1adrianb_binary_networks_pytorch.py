import sys
_module = sys.modules[__name__]
del sys
bnn = _module
bconfig = _module
binarize = _module
engine = _module
layers = _module
conv = _module
helpers = _module
linear = _module
models = _module
bats = _module
bats_ops = _module
common = _module
hierarchical_block = _module
res_block = _module
resnet = _module
ops = _module
version = _module
cifar10 = _module
imagenet = _module
utils = _module
setup = _module
smoke_test = _module
test_binarize = _module
test_engine = _module
test_layers = _module

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


import re


import copy


import logging


from typing import Dict


from typing import List


from typing import Callable


from typing import Union


from torch.nn.common_types import _size_1_t


from torch.nn.common_types import _size_2_t


import torch.nn.functional as F


from typing import Tuple


from collections import namedtuple


from typing import Optional


from typing import Type


from typing import Any


from functools import partial


from abc import ABCMeta


from abc import abstractmethod


import math


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torchvision


import torchvision.transforms as transforms


import time


import random


import warnings


import numpy as np


import torch.nn.parallel


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.datasets as datasets


class Identity(nn.Identity):

    def forward(self, layer_out: 'torch.Tensor', layer_in: 'torch.Tensor') ->torch.Tensor:
        return layer_out


def copy_paramters(source_mod: 'torch.nn.Module', target_mod: 'torch.nn.Module', bconfig: 'BConfig') ->None:
    attributes = [field.name for field in fields(bconfig)]
    for attribute in attributes:
        attr_obj_source = getattr(source_mod, attribute, None)
        attr_obj_target = getattr(target_mod, attribute, None)
        for name, source_param in attr_obj_source.named_parameters():
            if source_param is not None and attr_obj_target is not None:
                target_param = getattr(attr_obj_target, name, None)
                if target_param is not None:
                    if torch.equal(torch.tensor(target_param.size()), torch.tensor(source_param.size())):
                        target_param.data.copy_(source_param.data)


class Conv1d(nn.Conv1d):
    _FLOAT_MODULE = nn.Conv1d

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: '_size_1_t', stride: '_size_1_t'=1, padding: 'Union[str, _size_1_t]'=0, dilation: '_size_1_t'=1, groups: 'int'=1, bias: 'bool'=True, padding_mode: 'str'='zeros', bconfig: 'BConfig'=None) ->None:
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        assert bconfig, 'bconfig is required for a binarized module'
        self.bconfig = bconfig
        self.activation_pre_process = bconfig.activation_pre_process()
        self.activation_post_process = bconfig.activation_post_process(self)
        self.weight_pre_process = bconfig.weight_pre_process()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        input_proc = self.activation_pre_process(input)
        input_proc = self._conv_forward(input_proc, self.weight_pre_process(self.weight), bias=self.bias)
        return self.activation_post_process(input_proc, input)

    @classmethod
    def from_module(cls, mod: 'nn.Module', bconfig: 'BConfig'=None, update: 'bool'=False):
        assert type(mod) == cls._FLOAT_MODULE or type(mod) == cls, 'bnn.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        if not bconfig:
            assert hasattr(mod, 'bconfig'), 'The input modele requires a predifined bconfig'
            assert mod.bconfig, 'The input modele bconfig is invalid'
            bconfig = mod.bconfig
        bnn_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups, bias=mod.bias is not None, padding_mode=mod.padding_mode, bconfig=bconfig)
        bnn_conv.weight = mod.weight
        bnn_conv.bias = mod.bias
        if update:
            copy_paramters(mod, bnn_conv, bconfig)
        return bnn_conv


class Conv2d(nn.Conv2d):
    _FLOAT_MODULE = nn.Conv2d

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: '_size_2_t', stride: '_size_2_t'=1, padding: 'Union[str, _size_2_t]'=0, dilation: '_size_2_t'=1, groups: 'int'=1, bias: 'bool'=True, padding_mode: 'str'='zeros', bconfig: 'BConfig'=None) ->None:
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        assert bconfig, 'bconfig is required for a binarized module'
        self.bconfig = bconfig
        self.activation_pre_process = bconfig.activation_pre_process()
        self.activation_post_process = bconfig.activation_post_process(self)
        self.weight_pre_process = bconfig.weight_pre_process()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        input_proc = self.activation_pre_process(input)
        input_proc = self._conv_forward(input_proc, self.weight_pre_process(self.weight), bias=self.bias)
        return self.activation_post_process(input_proc, input)

    @classmethod
    def from_module(cls, mod: 'nn.Module', bconfig: 'BConfig'=None, update: 'bool'=False):
        assert type(mod) == cls._FLOAT_MODULE or type(mod) == cls, 'bnn.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        if not bconfig:
            assert hasattr(mod, 'bconfig'), 'The input modele requires a predifined bconfig'
            assert mod.bconfig, 'The input modele bconfig is invalid'
            bconfig = mod.bconfig
        bnn_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups, bias=mod.bias is not None, padding_mode=mod.padding_mode, bconfig=bconfig)
        bnn_conv.weight = mod.weight
        bnn_conv.bias = mod.bias
        if update:
            copy_paramters(mod, bnn_conv, bconfig)
        return bnn_conv


class Linear(nn.Linear):
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, bconfig: 'BConfig'=None) ->None:
        super(Linear, self).__init__(in_features, out_features, bias)
        assert bconfig, 'bconfig is required for a binarized module'
        self.bconfig = bconfig
        self.activation_pre_process = bconfig.activation_pre_process()
        self.activation_post_process = bconfig.activation_post_process(self)
        self.weight_pre_process = bconfig.weight_pre_process()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        input_proc = self.activation_pre_process(input)
        return self.activation_post_process(F.linear(input_proc, self.weight_pre_process(self.weight), self.bias), input)

    @classmethod
    def from_module(cls, mod: 'nn.Module', bconfig: 'BConfig'=None, update: 'bool'=False) ->nn.Module:
        assert type(mod) == cls._FLOAT_MODULE or type(mod) == cls, 'bnn.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        if not bconfig:
            assert hasattr(mod, 'bconfig'), 'The input modele requires a predifined bconfig'
            assert mod.bconfig, 'The input modele bconfig is invalid'
            bconfig = mod.bconfig
        bnn_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, bconfig=bconfig)
        bnn_linear.weight = mod.weight
        bnn_linear.bias = mod.bias
        if update:
            copy_paramters(mod, bnn_linear, bconfig)
        return bnn_linear


class FactorizedReduce(nn.Module):

    def __init__(self, C_in: 'int', C_out: 'int', affine: 'bool'=True) ->None:
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.activation = nn.PReLU(num_parameters=C_out)
        self.conv_1 = nn.Sequential(nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False))
        self.conv_2 = nn.Sequential(nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False))
        self.bn = nn.BatchNorm2d(C_in, affine=affine)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.bn(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.activation(out)
        return out


def channel_shuffle(x: 'torch.Tensor', groups: 'int') ->torch.Tensor:
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class DilConv(nn.Module):

    def __init__(self, C_in: 'int', C_out: 'int', kernel_size: 'int', stride: 'int', padding: 'int', dilation: 'int', affine: 'bool'=True, skip: 'bool'=False, groups: 'int'=12) ->None:
        super(DilConv, self).__init__()
        self.skip = skip or True
        self.stride = stride
        self.op = nn.Sequential(nn.BatchNorm2d(C_in, affine=affine), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False), nn.PReLU(num_parameters=C_in))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.skip and self.stride == 1:
            return x + channel_shuffle(self.op(x), 4)
        else:
            return channel_shuffle(self.op(x), 4)


class FactorizedConv(nn.Module):

    def __init__(self, C: 'int', kernel_size: 'int', stride: 'int', affine: 'bool'=True, skip: 'bool'=False) ->None:
        super(FactorizedConv, self).__init__()
        self.skip = skip or True
        self.stride = stride
        self.op = nn.Sequential(nn.BatchNorm2d(C, affine=affine), nn.Conv2d(C, C, (1, kernel_size), stride=(1, stride), padding=(0, kernel_size // 2), bias=False), nn.PReLU(num_parameters=C), nn.BatchNorm2d(C, affine=affine), nn.Conv2d(C, C, (kernel_size, 1), stride=(stride, 1), padding=(kernel_size // 2, 0), bias=False), nn.PReLU(num_parameters=C))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.skip and self.stride == 1:
            return x + channel_shuffle(self.op(x), 4)
        else:
            return channel_shuffle(self.op(x), 4)


class SepConv(nn.Module):

    def __init__(self, C_in: 'int', C_out: 'int', kernel_size: 'int', stride: 'int', padding: 'int', affine: 'bool'=True, skip: 'bool'=False, groups: 'int'=12) ->None:
        super(SepConv, self).__init__()
        self.skip = skip or True
        self.stride = stride
        self.op = nn.Sequential(nn.BatchNorm2d(C_in, affine=affine), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False), nn.PReLU(num_parameters=C_in))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.skip and self.stride == 1:
            return x + channel_shuffle(self.op(x), 4)
        else:
            return channel_shuffle(self.op(x), 4)


class Zero(nn.Module):

    def __init__(self, stride: 'int') ->None:
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        padding = torch.zeros(n, c, h, w, dtype=torch.float32, device=x.device)
        return padding


OPS = {'none': lambda C, stride, affine, skip, groups: Zero(stride), 'avg_pool_3x3': lambda C, stride, affine, skip, groups: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), 'max_pool_3x3': lambda C, stride, affine, skip, groups: nn.MaxPool2d(3, stride=stride, padding=1), 'skip_connect': lambda C, stride, affine, skip, groups: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine), 'sep_conv_3x3': lambda C, stride, affine, skip, groups: SepConv(C, C, 3, stride, 1, affine=affine, skip=skip, groups=groups), 'sep_conv_5x5': lambda C, stride, affine, skip, groups: SepConv(C, C, 5, stride, 2, affine=affine, skip=skip, groups=groups), 'sep_conv_7x7': lambda C, stride, affine, skip, groups: SepConv(C, C, 7, stride, 3, affine=affine, skip=skip, groups=groups), 'dil_conv_3x3': lambda C, stride, affine, skip, groups: DilConv(C, C, 3, stride, 2, 2, affine=affine, skip=skip, groups=groups), 'dil_conv_5x5': lambda C, stride, affine, skip, groups: DilConv(C, C, 5, stride, 4, 2, affine=affine, skip=skip, groups=groups), 'conv_7x1_1x7': lambda C, stride, affine, skip, groups: FactorizedConv(C, 7, stride, affine=affine, skip=skip)}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in: 'int', C_out: 'int', kernel_size: 'int', stride: 'int', padding: 'int', affine: 'bool'=True, skip: 'bool'=False) ->None:
        super(ReLUConvBN, self).__init__()
        self.skip = skip or True
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.op = nn.Sequential(nn.BatchNorm2d(C_in, affine=affine), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.PReLU(num_parameters=C_out))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.skip and self.stride == 1 and self.C_in == self.C_out:
            return x + self.op(x)
        else:
            return self.op(x)


def drop_path(x: 'torch.Tensor', drop_prob: 'float') ->torch.Tensor:
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.tensor(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype: 'Genotype', C_prev_prev: 'int', C_prev: 'int', C: 'int', reduction: 'bool', reduction_prev: 'bool', groups: 'int'=12, use_shake_shake: 'bool'=False) ->None:
        super(Cell, self).__init__()
        self.use_shake_shake = use_shake_shake
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction, groups)

    def _compile(self, C: 'int', op_names: 'List[str]', indices: 'List[int]', concat: 'List[int]', reduction: 'bool', groups: 'int') ->None:
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, True, groups)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0: 'torch.Tensor', s1: 'torch.Tensor', drop_prob: 'float') ->torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, nn.Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, nn.Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        if self.use_shake_shake:
            if self.training:
                shake = torch.softmax(torch.zeros(len(self._concat)).uniform_(), dim=0)
                return torch.cat([(states[i] * shake[j].item()) for j, i in enumerate(self._concat)], dim=1)
            else:
                return torch.cat([(states[i] * (1 / len(self._concat))) for i in self._concat], dim=1)
        else:
            return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHead(nn.Module):

    def __init__(self, C: 'int', num_classes: 'int', stride: 'int') ->None:
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False), nn.BatchNorm2d(C), nn.Conv2d(C, 128, 1, bias=False), nn.PReLU(num_parameters=128), nn.BatchNorm2d(128), nn.Conv2d(128, 768, 2, bias=False), nn.PReLU(num_parameters=768))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class BATSNetworkCIFAR(nn.Module):

    def __init__(self, C: 'int', num_classes: 'int', layers: 'int', auxiliary: 'bool', genotype, groups: 'int') ->None:
        super(BATSNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr), nn.ReLU(inplace=True))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, groups)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes, 3)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class BATSNetworkImageNet(nn.Module):

    def __init__(self, C: 'int', num_classes: 'int', layers: 'int', auxiliary: 'bool', genotype, groups: 'int') ->None:
        super(BATSNetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.stem0 = nn.Sequential(nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C // 2), nn.ReLU(inplace=True), nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False, groups=C // 20), nn.BatchNorm2d(C))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False, groups=C // 20), nn.BatchNorm2d(C))
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, groups)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes, 2)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


def conv3x3(in_planes: 'int', out_planes: 'int', stride: 'int'=1, groups: 'int'=1, dilation: 'int'=1) ->nn.Module:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class HBlock(nn.Module):

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation=nn.ReLU) ->None:
        super(HBlock, self).__init__()
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in HBlock')
        if stride > 1:
            raise NotImplementedError('Stride > 1 not supported in HBlock')
        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv3x3(inplanes, int(planes / 2), groups=groups)
        self.bn2 = norm_layer(int(planes / 2))
        self.conv2 = conv3x3(int(planes / 2), int(planes / 4), groups=groups)
        self.bn3 = norm_layer(int(planes / 4))
        self.conv3 = conv3x3(int(planes / 4), int(planes / 4), groups=groups)
        self.act1 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=int(planes / 2))
        self.act2 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=int(planes / 2))
        self.act3 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=int(planes / 4))
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = self.act1(out1)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = self.act2(out2)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = self.act3(out3)
        out3 = self.conv3(out3)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation=nn.ReLU) ->None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.act1 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=planes)
        self.act2 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act2(out)
        return out


def conv1x1(in_planes: 'int', out_planes: 'int', stride: 'int'=1) ->nn.Module:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation=nn.ReLU) ->None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act1 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=width)
        self.act2 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=width)
        self.act3 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act3(out)
        return out


class PreBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation=nn.ReLU) ->None:
        super(PreBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.act1 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=planes)
        self.act2 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.act2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class PreBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation=nn.ReLU) ->None:
        super(PreBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(width)
        self.act1 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=width)
        self.act2 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=width)
        self.act3 = activation(inplace=True) if activation == nn.ReLU else activation(num_parameters=planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.bn3(out)
        out = self.conv3(out)
        out = self.act3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class DaBNNStem(nn.Module):

    def __init__(self, planes: 'int', norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation=nn.ReLU):
        super(DaBNNStem, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, planes // 2, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(planes // 2), activation())
        self.conv2_1 = nn.Sequential(nn.Conv2d(planes // 2, planes // 4, 1, 1, bias=False), norm_layer(planes // 4), activation())
        self.conv2_2 = nn.Sequential(nn.Conv2d(planes // 4, planes // 2, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(planes // 2), activation())
        self.conv3 = nn.Sequential(nn.Conv2d(planes, planes, 1, 1, bias=False), norm_layer(planes), activation())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([self.conv2_2(self.conv2_1(x)), self.maxpool(x)], dim=1)
        x = self.conv3(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block: 'Type[Union[BasicBlock, Bottleneck, HBlock, PreBasicBlock, PreBottleneck]]', layers: 'List[int]', num_classes: 'int'=1000, zero_init_residual: 'bool'=False, groups: 'int'=1, width_per_group: 'int'=64, replace_stride_with_dilation: 'Optional[List[bool]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation: 'Optional[Callable[..., nn.Module]]'=None, stem_type: 'str'='basic') ->None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU
        self._norm_layer = norm_layer
        self._activation = activation
        self.stem_type = stem_type
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if stem_type == 'basic':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
        elif stem_type == 'dabnn':
            self.conv1 = DaBNNStem(self.inplanes, norm_layer=norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.outplanes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: 'Type[Union[BasicBlock, Bottleneck]]', planes: 'int', blocks: 'int', stride: 'int'=1, dilate: 'bool'=False) ->nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False), conv1x1(self.inplanes, planes * block.expansion, stride=1), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, activation=self._activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, activation=self._activation))
        self.outplanes = planes
        return nn.Sequential(*layers)

    def _forward_impl(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv1(x)
        if self.stem_type == 'basic':
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self._forward_impl(x)


ABC = ABCMeta(str('ABC'), (object,), {})


def _with_args(cls_or_self: 'Any', **kwargs: Dict[str, Any]) ->Any:
    """Wrapper that allows creation of class factories.
    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.
    Source: https://github.com/pytorch/pytorch/blob/b02c932fb67717cb26d6258908541b670faa4e72/torch/quantization/observer.py
    Example::
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """


    class _PartialWrapper(object):

        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()
        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


class BinarizerBase(ABC, nn.Module):

    def __init__(self) ->None:
        super(BinarizerBase, self).__init__()

    @abstractmethod
    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        pass
    with_args = classmethod(_with_args)


class SignActivation(torch.autograd.Function):
    """Applies the sign function element-wise
    :math:`\\text{sgn(x)} = \\begin{cases} -1 & \\text{if } x < 0, \\\\ 1 & \\text{if} x >0  \\end{cases}`
    the gradients of which are computed using a STE, namely using :math:`\\text{hardtanh(x)}`.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> input = torch.randn(3)
        >>> output = SignActivation.apply(input)
    """

    @staticmethod
    def forward(ctx, input: 'torch.Tensor') ->torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor') ->torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input


class XNORWeightBinarizer(BinarizerBase):
    """Binarize the parameters of a given layer using the analytical solution
    proposed in the XNOR-Net paper.
    :math:`\\text{out} = \\frac{1}{n}\\norm{\\mathbf{W}}_{\\ell} \\text{sgn(x)}(\\mathbf{W})`
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> binarizer = XNORWeightBinarizer()
        >>> output = F.conv2d(input, binarizer(weight))
    Args:
        compute_alpha: if True, compute the real-valued scaling factor
        center_weights: make the weights zero-mean
    """

    def __init__(self, compute_alpha: 'bool'=True, center_weights: 'bool'=False) ->None:
        super(XNORWeightBinarizer, self).__init__()
        self.compute_alpha = compute_alpha
        self.center_weights = center_weights

    def _compute_alpha(self, x: 'torch.Tensor') ->torch.Tensor:
        n = x[0].nelement()
        if x.dim() == 4:
            alpha = x.norm(1, 3, keepdim=True).sum([2, 1], keepdim=True).div_(n)
        elif x.dim() == 3:
            alpha = x.norm(1, 2, keepdim=True).sum([1], keepdim=True).div_(n)
        elif x.dim() == 2:
            alpha = x.norm(1, 1, keepdim=True).div_(n)
        else:
            raise ValueError(f'Expected ndims equal with 2 or 4, but found {x.dim()}')
        return alpha

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.center_weights:
            mean = x.mean(1, keepdim=True).expand_as(x)
            x = x.sub(mean)
        if self.compute_alpha:
            alpha = self._compute_alpha(x)
            x = SignActivation.apply(x).mul_(alpha.expand_as(x))
        else:
            x = SignActivation.apply(x)
        return x


class BasicInputBinarizer(BinarizerBase):
    """Applies the sign function element-wise.
    nn.Module version of SignActivation.
    """

    def __init__(self):
        super(BasicInputBinarizer, self).__init__()

    def forward(self, x: 'torch.Tensor') ->None:
        return SignActivation.apply(x)


class SignActivationStochastic(SignActivation):
    """Binarize the data using a stochastic binarizer
    :math:`\\text{sgn(x)} = \\begin{cases} -1 & \\text{with probablity } p = \\sigma(x), \\\\ 1 & \\text{with probablity } 1 - p \\end{cases}`
    the gradients of which are computed using a STE, namely using :math:`\\text{hardtanh(x)}`.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> input = torch.randn(3)
        >>> output = SignActivationStochastic.apply(input)
    """

    @staticmethod
    def forward(ctx, input: 'torch.Tensor') ->torch.Tensor:
        ctx.save_for_backward(input)
        noise = torch.rand_like(input).sub_(0.5)
        return input.add_(1).div_(2).add_(noise).clamp_(0, 1).round_().mul_(2).sub_(1)


class StochasticInputBinarizer(BinarizerBase):
    """Applies the sign function element-wise.
    nn.Module version of SignActivation.
    """

    def __init__(self):
        super(StochasticInputBinarizer, self).__init__()

    def forward(self, x: 'torch.Tensor'):
        return SignActivationStochastic.apply(x)


class AdvancedInputBinarizer(BinarizerBase):

    def __init__(self, derivative_funct=torch.tanh, t: 'int'=5):
        super(AdvancedInputBinarizer, self).__init__()
        self.derivative_funct = derivative_funct
        self.t = t

    def forward(self, x: 'torch.tensor') ->torch.Tensor:
        x = self.derivative_funct(x * self.t)
        with torch.no_grad():
            x = torch.sign(x)
        return x


class BasicScaleBinarizer(BinarizerBase):

    def __init__(self, module: 'nn.Module', shape: 'List[int]'=None) ->None:
        super(BasicScaleBinarizer, self).__init__()
        if isinstance(module, nn.Linear):
            num_channels = module.out_features
        elif isinstance(module, nn.Conv2d):
            num_channels = module.out_channels
        elif hasattr(module, 'out_channels'):
            num_channels = module.out_channels
        else:
            raise Exception('Unknown layer of type {} missing out_channels'.format(type(module)))
        if shape is None:
            alpha_shape = [1, num_channels] + [1] * (module.weight.dim() - 2)
        else:
            alpha_shape = shape
        self.alpha = nn.Parameter(torch.ones(*alpha_shape))

    def forward(self, layer_out: 'torch.Tensor', layer_in: 'torch.Tensor') ->torch.Tensor:
        x = layer_out
        return x.mul_(self.alpha)

    def extra_repr(self) ->str:
        return '{}'.format(list(self.alpha.size()))


class XNORScaleBinarizer(BinarizerBase):

    def __init__(self, module: 'nn.Module') ->None:
        super(BasicScaleBinarizer, self).__init__()
        kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.register_buffer('fixed_weight', torch.ones(*kernel_size).div_(math.prod(kernel_size)), persistent=False)

    def forward(self, layer_out: 'torch.Tensor', layer_in: 'torch.Tensor') ->torch.Tensor:
        x = layer_out
        scale = torch.mean(dim=1, keepdim=True)
        scale = F.conv2d(scale, self.fixed_weight, stride=self.stride, padding=self.padding)
        return x.mul_(scale)


class Flatten(nn.Module):

    def __init__(self) ->None:
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdvancedInputBinarizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicInputBinarizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FactorizedReduce,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StochasticInputBinarizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (XNORWeightBinarizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Zero,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_1adrianb_binary_networks_pytorch(_paritybench_base):
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

