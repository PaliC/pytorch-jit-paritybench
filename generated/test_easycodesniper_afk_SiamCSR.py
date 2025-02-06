import sys
_module = sys.modules[__name__]
del sys
SiamCSRTracker = _module
SiamCSR = _module
attention = _module
bbox_util = _module
config = _module
loss = _module
rgbt_network = _module
transforms = _module
utils = _module
my_demo = _module
my_test_rgbt = _module
my_train = _module
data = _module
datawash = _module
gtot = _module
rgbt234 = _module
rgbt234_lasher = _module
rgbt_dataset = _module

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


import torch


import torch.nn.functional as F


import time


import torchvision.transforms as transforms


from torch import nn


from torch.nn import init


import torch.nn


import random


import functools


from torch.multiprocessing import Pool


from torch.multiprocessing import Manager


import math


import matplotlib.pyplot as plt


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


class LocalAttentionBlock(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, rgb_feature, t_feature):
        shape = torch.cat((rgb_feature, t_feature), 1).shape
        union_feature = torch.randn(shape[0], shape[1], shape[2], shape[3])
        i, idx1, idx2 = 0, 0, 0
        while i < shape[1]:
            if i % 2 == 0:
                union_feature[:, i, :, :] = rgb_feature[:, idx1, :, :]
                idx1 += 1
            else:
                union_feature[:, i, :, :] = t_feature[:, idx2, :, :]
                idx2 += 1
            i += 1
        y = self.gap(union_feature)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        union_feature = union_feature * y.expand_as(union_feature)
        i = 0
        while i < shape[1]:
            if i % 2 == 0:
                rgb_feature[:, i // 2, :, :] = union_feature[:, i, :, :]
            else:
                t_feature[:, i // 2, :, :] = union_feature[:, i, :, :]
            i += 1
        return rgb_feature, t_feature


class GlobalAttentionBlock(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, rgb_feature, t_feature):
        channel_num = rgb_feature.shape[1]
        union_feature = torch.cat((rgb_feature, t_feature), 1)
        b, c, _, _ = union_feature.size()
        y = self.avg_pool(union_feature).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        union_feature = union_feature * y.expand_as(union_feature)
        return union_feature[:, :channel_num, :, :], union_feature[:, channel_num:, :, :]


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate_rgb = SpatialGate()
            self.SpatialGate_t = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out_rgb = x_out[:, :256, :, :]
        x_out_t = x_out[:, 256:, :, :]
        if not self.no_spatial:
            x_out_rgb = self.SpatialGate_rgb(x_out_rgb)
            x_out_t = self.SpatialGate_t(x_out_t)
        return x_out_rgb, x_out_t


class SiamCSRNet(nn.Module):

    def __init__(self):
        super(SiamCSRNet, self).__init__()
        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)
        self.former_3_layers_featureExtract = nn.Sequential(nn.Conv2d(3, 96, 11, stride=2), nn.BatchNorm2d(96), nn.MaxPool2d(3, stride=2), nn.ReLU(inplace=True), nn.Conv2d(96, 256, 5), nn.BatchNorm2d(256), nn.MaxPool2d(3, stride=2), nn.ReLU(inplace=True), nn.Conv2d(256, 384, 3), nn.BatchNorm2d(384), nn.ReLU(inplace=True))
        self.rgb_featureExtract = nn.Sequential(nn.Conv2d(384, 384, 3), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3), nn.BatchNorm2d(256))
        self.t_featureExtract = nn.Sequential(nn.Conv2d(384, 384, 3), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3), nn.BatchNorm2d(256))
        self.rgb_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.rgb_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)
        self.t_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.t_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.t_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.t_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.t_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)
        self.attn_rgb_featureExtract = nn.Sequential(nn.Conv2d(384, 384, 3), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3), nn.BatchNorm2d(256))
        self.attn_t_featureExtract = nn.Sequential(nn.Conv2d(384, 384, 3), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3), nn.BatchNorm2d(256))
        self.attn_rgb_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)
        self.attn_t_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_t_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)
        self.template_attention_block = GlobalAttentionBlock()
        self.detection_attention_block = CBAM(512)

    def forward(self, rgb_template, rgb_detection, t_template, t_detection):
        N = rgb_template.size(0)
        rgb_template = self.former_3_layers_featureExtract(rgb_template)
        rgb_detection = self.former_3_layers_featureExtract(rgb_detection)
        t_template = self.former_3_layers_featureExtract(t_template)
        t_detection = self.former_3_layers_featureExtract(t_detection)
        rgb_template_feature = self.rgb_featureExtract(rgb_template)
        rgb_detection_feature = self.rgb_featureExtract(rgb_detection)
        attn_rgb_template_feature = self.attn_rgb_featureExtract(rgb_template)
        attn_rgb_detection_feature = self.attn_rgb_featureExtract(rgb_detection)
        t_template_feature = self.t_featureExtract(t_template)
        t_detection_feature = self.t_featureExtract(t_detection)
        attn_t_template_feature = self.attn_t_featureExtract(t_template)
        attn_t_detection_feature = self.attn_t_featureExtract(t_detection)
        attn_rgb_template_feature, attn_t_template_feature = self.template_attention_block(attn_rgb_template_feature, attn_t_template_feature)
        union = torch.cat((attn_rgb_detection_feature, attn_t_detection_feature), 1)
        attn_rgb_detection_feature, attn_t_detection_feature = self.detection_attention_block(union)
        rgb_kernel_score = self.rgb_conv_cls1(rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        rgb_kernel_regression = self.rgb_conv_r1(rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        rgb_conv_score = self.rgb_conv_cls2(rgb_detection_feature)
        rgb_conv_regression = self.rgb_conv_r2(rgb_detection_feature)
        rgb_conv_scores = rgb_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        rgb_score_filters = rgb_kernel_score.reshape(-1, 256, 4, 4)
        rgb_pred_score = F.conv2d(rgb_conv_scores, rgb_score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        rgb_conv_reg = rgb_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        rgb_reg_filters = rgb_kernel_regression.reshape(-1, 256, 4, 4)
        rgb_pred_regression = self.rgb_regress_adjust(F.conv2d(rgb_conv_reg, rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        attn_rgb_kernel_score = self.attn_rgb_conv_cls1(attn_rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_rgb_kernel_regression = self.attn_rgb_conv_r1(attn_rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        attn_rgb_conv_score = self.attn_rgb_conv_cls2(attn_rgb_detection_feature)
        attn_rgb_conv_regression = self.attn_rgb_conv_r2(attn_rgb_detection_feature)
        attn_rgb_conv_scores = attn_rgb_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_rgb_score_filters = attn_rgb_kernel_score.reshape(-1, 256, 4, 4)
        attn_rgb_pred_score = F.conv2d(attn_rgb_conv_scores, attn_rgb_score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        attn_rgb_conv_reg = attn_rgb_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_rgb_reg_filters = attn_rgb_kernel_regression.reshape(-1, 256, 4, 4)
        attn_rgb_pred_regression = self.attn_rgb_regress_adjust(F.conv2d(attn_rgb_conv_reg, attn_rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        t_kernel_score = self.t_conv_cls1(t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        t_conv_score = self.t_conv_cls2(t_detection_feature)
        t_conv_scores = t_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        t_score_filters = t_kernel_score.reshape(-1, 256, 4, 4)
        t_pred_score = F.conv2d(t_conv_scores, t_score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        t_kernel_regression = self.t_conv_r1(t_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        t_reg_filters = t_kernel_regression.reshape(-1, 256, 4, 4)
        t_conv_regression = self.t_conv_r2(t_detection_feature)
        t_conv_reg = t_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        t_pred_regression = self.t_regress_adjust(F.conv2d(t_conv_reg, t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        attn_t_kernel_score = self.attn_t_conv_cls1(attn_t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_t_conv_score = self.attn_t_conv_cls2(attn_t_detection_feature)
        attn_t_conv_scores = attn_t_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_t_score_filters = attn_t_kernel_score.reshape(-1, 256, 4, 4)
        attn_t_pred_score = F.conv2d(attn_t_conv_scores, attn_t_score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        attn_t_kernel_regression = self.attn_t_conv_r1(attn_t_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        attn_t_reg_filters = attn_t_kernel_regression.reshape(-1, 256, 4, 4)
        attn_t_conv_regression = self.attn_t_conv_r2(attn_t_detection_feature)
        attn_t_conv_reg = attn_t_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_t_pred_regression = self.attn_t_regress_adjust(F.conv2d(attn_t_conv_reg, attn_t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        return rgb_pred_score + attn_t_pred_score, attn_rgb_pred_score + t_pred_score, rgb_pred_regression + attn_t_pred_regression, attn_rgb_pred_regression + t_pred_regression

    def track_init(self, rgb_template, t_template):
        N = rgb_template.size(0)
        rgb_template = self.former_3_layers_featureExtract(rgb_template)
        t_template = self.former_3_layers_featureExtract(t_template)
        rgb_template_feature = self.rgb_featureExtract(rgb_template)
        t_template_feature = self.t_featureExtract(t_template)
        attn_rgb_template_feature = self.attn_rgb_featureExtract(rgb_template)
        attn_t_template_feature = self.attn_t_featureExtract(t_template)
        attn_rgb_template_feature, attn_t_template_feature = self.template_attention_block(attn_rgb_template_feature, attn_t_template_feature)
        rgb_kernel_score = self.rgb_conv_cls1(rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        t_kernel_score = self.t_conv_cls1(t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        self.rgb_score_filters = rgb_kernel_score.reshape(-1, 256, 4, 4)
        self.t_score_filters = t_kernel_score.reshape(-1, 256, 4, 4)
        rgb_kernel_regression = self.rgb_conv_r1(rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        t_kernel_regression = self.t_conv_r1(t_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.rgb_reg_filters = rgb_kernel_regression.reshape(-1, 256, 4, 4)
        self.t_reg_filters = t_kernel_regression.reshape(-1, 256, 4, 4)
        attn_rgb_kernel_score = self.attn_rgb_conv_cls1(attn_rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_t_kernel_score = self.attn_t_conv_cls1(attn_t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        self.attn_rgb_score_filters = attn_rgb_kernel_score.reshape(-1, 256, 4, 4)
        self.attn_t_score_filters = attn_t_kernel_score.reshape(-1, 256, 4, 4)
        attn_rgb_kernel_regression = self.attn_rgb_conv_r1(attn_rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        attn_t_kernel_regression = self.attn_t_conv_r1(attn_t_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.attn_rgb_reg_filters = attn_rgb_kernel_regression.reshape(-1, 256, 4, 4)
        self.attn_t_reg_filters = attn_t_kernel_regression.reshape(-1, 256, 4, 4)

    def track(self, rgb_detection, t_detection):
        N = rgb_detection.size(0)
        rgb_detection = self.former_3_layers_featureExtract(rgb_detection)
        t_detection = self.former_3_layers_featureExtract(t_detection)
        rgb_detection_feature = self.rgb_featureExtract(rgb_detection)
        t_detection_feature = self.t_featureExtract(t_detection)
        attn_rgb_detection_feature = self.attn_rgb_featureExtract(rgb_detection)
        attn_t_detection_feature = self.attn_t_featureExtract(t_detection)
        union = torch.cat((attn_rgb_detection_feature, attn_t_detection_feature), 1)
        attn_rgb_detection_feature, attn_t_detection_feature = self.detection_attention_block(union)
        rgb_conv_score = self.rgb_conv_cls2(rgb_detection_feature)
        t_conv_score = self.t_conv_cls2(t_detection_feature)
        rgb_conv_regression = self.rgb_conv_r2(rgb_detection_feature)
        t_conv_regression = self.t_conv_r2(t_detection_feature)
        rgb_conv_scores = rgb_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        rgb_pred_score = F.conv2d(rgb_conv_scores, self.rgb_score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        t_conv_scores = t_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        t_pred_score = F.conv2d(t_conv_scores, self.t_score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        rgb_conv_reg = rgb_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        t_conv_reg = t_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        rgb_pred_regression = self.rgb_regress_adjust(F.conv2d(rgb_conv_reg, self.rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        t_pred_regression = self.t_regress_adjust(F.conv2d(t_conv_reg, self.t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        attn_rgb_conv_score = self.attn_rgb_conv_cls2(attn_rgb_detection_feature)
        attn_t_conv_score = self.attn_t_conv_cls2(attn_t_detection_feature)
        attn_rgb_conv_regression = self.attn_rgb_conv_r2(attn_rgb_detection_feature)
        attn_t_conv_regression = self.attn_t_conv_r2(attn_t_detection_feature)
        attn_rgb_conv_scores = attn_rgb_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_rgb_pred_score = F.conv2d(attn_rgb_conv_scores, self.attn_rgb_score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        attn_t_conv_scores = attn_t_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_t_pred_score = F.conv2d(attn_t_conv_scores, self.attn_t_score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        attn_rgb_conv_reg = attn_rgb_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_t_conv_reg = attn_t_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_rgb_pred_regression = self.attn_rgb_regress_adjust(F.conv2d(attn_rgb_conv_reg, self.attn_rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        attn_t_pred_regression = self.attn_t_regress_adjust(F.conv2d(attn_t_conv_reg, self.attn_t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        return rgb_pred_score + attn_t_pred_score, attn_rgb_pred_score + t_pred_score, rgb_pred_regression + t_pred_regression


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelGate,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocalAttentionBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_easycodesniper_afk_SiamCSR(_paritybench_base):
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

