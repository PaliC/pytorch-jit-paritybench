import sys
_module = sys.modules[__name__]
del sys
dataset = _module
kitti_dataset = _module
model = _module
correlation_package = _module
correlation = _module
setup = _module
pwc_modules = _module
upflow = _module
scripts = _module
ex_runner = _module
simple_train = _module
test = _module
utils = _module
loss = _module
pytorch_correlation = _module
tools = _module

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


import random


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import numpy as np


import torch


import tensorflow as tf


import warnings


from torchvision import transforms as vision_transforms


from torch.nn.modules.module import Module


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.nn as nn


import torch.nn.functional as tf


import logging


import collections


import torch.nn.functional as F


import math


from copy import deepcopy


import torch.optim as optim


import time


from torch.utils.data.dataloader import DataLoader


import torch.utils.model_zoo as model_zoo


from torch.nn.init import xavier_normal


from torch.nn.init import kaiming_normal


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)
        return output

    def backward(self, grad_output):
        input1, input2 = self.saved_tensors
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)
        return grad_input1, grad_input2


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)
        return result


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False, if_BN=False):
    if isReLU:
        if if_IN:
            return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=True), nn.LeakyReLU(0.1, inplace=True), nn.InstanceNorm2d(out_planes, affine=IN_affine))
        elif if_BN:
            return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=True), nn.LeakyReLU(0.1, inplace=True), nn.BatchNorm2d(out_planes, affine=IN_affine))
        else:
            return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=True), nn.LeakyReLU(0.1, inplace=True))
    elif if_IN:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=True), nn.InstanceNorm2d(out_planes, affine=IN_affine))
    elif if_BN:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=True), nn.BatchNorm2d(out_planes, affine=IN_affine))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) * dilation // 2, bias=True))


class FeatureExtractor(nn.Module):

    def __init__(self, num_chs, if_end_relu=True, if_end_norm=False):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()
        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(conv(ch_in, ch_out, stride=2), conv(ch_out, ch_out, isReLU=if_end_relu, if_IN=if_end_norm))
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)
        return feature_pyramid[::-1]


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    if x.is_cuda:
        grids_cuda = grid.float().requires_grad_(False)
    else:
        grids_cuda = grid.float().requires_grad_(False)
    return grids_cuda


class WarpingLayer(nn.Module):

    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid)
        if x.is_cuda:
            mask = torch.ones(x.size(), requires_grad=False)
        else:
            mask = torch.ones(x.size(), requires_grad=False)
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()
        return x_warp * mask


class WarpingLayer_no_div(nn.Module):

    def __init__(self):
        super(WarpingLayer_no_div, self).__init__()

    def forward(self, x, flow):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        x_warp = tf.grid_sample(x, vgrid, padding_mode='zeros')
        if x.is_cuda:
            mask = torch.ones(x.size(), requires_grad=False)
        else:
            mask = torch.ones(x.size(), requires_grad=False)
        mask = tf.grid_sample(mask, vgrid)
        mask = (mask >= 1.0).float()
        return x_warp * mask


class OpticalFlowEstimator(nn.Module):

    def __init__(self, ch_in):
        super(OpticalFlowEstimator, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128), conv(128, 128), conv(128, 96), conv(96, 64), conv(64, 32))
        self.conv_last = conv(32, 2, isReLU=False)

    def forward(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class FlowEstimatorDense(nn.Module):

    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class OcclusionEstimator(nn.Module):

    def __init__(self, ch_in):
        super(OcclusionEstimator, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128), conv(128, 128), conv(128, 96), conv(96, 64), conv(64, 32))
        self.conv_last = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class OccEstimatorDense(nn.Module):

    def __init__(self, ch_in):
        super(OccEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 1, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class ContextNetwork(nn.Module):

    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 2), conv(128, 128, 3, 1, 4), conv(128, 96, 3, 1, 8), conv(96, 64, 3, 1, 16), conv(64, 32, 3, 1, 1), conv(32, 2, isReLU=False))

    def forward(self, x):
        return self.convs(x)


class ContextNetwork_v2_(nn.Module):

    def __init__(self, ch_in, f_channels=(128, 128, 128, 96, 64, 32, 2)):
        super(ContextNetwork_v2_, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, f_channels[0], 3, 1, 1), conv(f_channels[0], f_channels[1], 3, 1, 2), conv(f_channels[1], f_channels[2], 3, 1, 4), conv(f_channels[2], f_channels[3], 3, 1, 8), conv(f_channels[3], f_channels[4], 3, 1, 16), conv(f_channels[4], f_channels[5], 3, 1, 1), conv(f_channels[5], f_channels[6], isReLU=False))

    def forward(self, x):
        return self.convs(x)


class ContextNetwork_v2(nn.Module):

    def __init__(self, num_ls=(3, 128, 128, 128, 96, 64, 32, 16)):
        super(ContextNetwork_v2, self).__init__()
        self.num_ls = num_ls
        cnt = 0
        cnt_in = num_ls[0]
        self.cov1 = conv(num_ls[0], num_ls[1], 3, 1, 1)
        cnt += 1
        cnt_in += num_ls[cnt]
        self.cov2 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 2)
        cnt += 1
        cnt_in += num_ls[cnt]
        self.cov3 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 4)
        cnt += 1
        cnt_in += num_ls[cnt]
        self.cov4 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 8)
        cnt += 1
        cnt_in += num_ls[cnt]
        self.cov5 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 16)
        cnt += 1
        cnt_in += num_ls[cnt]
        self.cov6 = conv(cnt_in, num_ls[cnt + 1], 3, 1, 1)
        cnt += 1
        cnt_in += num_ls[cnt]
        self.final = conv(cnt_in, num_ls[cnt + 1], isReLU=False)

    def forward(self, x):
        x = torch.cat((self.cov1(x), x), dim=1)
        x = torch.cat((self.cov2(x), x), dim=1)
        x = torch.cat((self.cov3(x), x), dim=1)
        x = torch.cat((self.cov4(x), x), dim=1)
        x = torch.cat((self.cov5(x), x), dim=1)
        x = torch.cat((self.cov6(x), x), dim=1)
        x = self.final(x)
        return x


class OccContextNetwork(nn.Module):

    def __init__(self, ch_in):
        super(OccContextNetwork, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 2), conv(128, 128, 3, 1, 4), conv(128, 96, 3, 1, 8), conv(96, 64, 3, 1, 16), conv(64, 32, 3, 1, 1), conv(32, 1, isReLU=False))

    def forward(self, x):
        return self.convs(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContextNetwork,
     lambda: ([], {'ch_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextNetwork_v2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ContextNetwork_v2_,
     lambda: ([], {'ch_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureExtractor,
     lambda: ([], {'num_chs': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FlowEstimatorDense,
     lambda: ([], {'ch_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OccContextNetwork,
     lambda: ([], {'ch_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OccEstimatorDense,
     lambda: ([], {'ch_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OcclusionEstimator,
     lambda: ([], {'ch_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OpticalFlowEstimator,
     lambda: ([], {'ch_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WarpingLayer_no_div,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 2, 4, 4])], {}),
     False),
]

class Test_coolbeam_UPFlow_pytorch(_paritybench_base):
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

