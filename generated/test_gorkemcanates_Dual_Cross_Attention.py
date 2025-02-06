import sys
_module = sys.modules[__name__]
del sys
Datasets = _module
loss = _module
main = _module
metrics = _module
double_unet = _module
multi_res_unet = _module
r2_unet = _module
res_unet_plus = _module
swin_unet = _module
transunet = _module
unet = _module
unext = _module
dca = _module
dca_utils = _module
main_blocks = _module
vnet = _module
trainer = _module
custom_transforms = _module
transforms = _module
writer = _module

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


import matplotlib.pyplot as plt


import random


import numpy as np


from sklearn.model_selection import train_test_split


import torchvision.transforms as transforms


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


from torch.optim import Adam


import time


from torch import nn


from torchvision.models import vgg19


import copy


import logging


import math


from torch.nn import CrossEntropyLoss


from torch.nn import Dropout


from torch.nn import Softmax


from torch.nn import Linear


from torch.nn import Conv2d


from torch.nn import LayerNorm


from torch.nn.modules.utils import _pair


from scipy import ndimage


import torch.utils.checkpoint as checkpoint


import torchvision


from torch.autograd import Variable


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.utils import save_image


import types


import warnings


import torch.multiprocessing as mp


from typing import Iterable


from typing import Any


import torchvision.transforms.functional as TF


from torch.utils.tensorboard import SummaryWriter


import matplotlib


class DiceLoss(nn.Module):

    def __init__(self, num_classes=1) ->None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.0

    def forward(self, pred, target):
        target = target.squeeze(1)
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            loss = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                loss += 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss /= self.num_classes
        return loss


class DiceScore(nn.Module):

    def __init__(self, num_classes=1) ->None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.0

    def forward(self, pred, target):
        target = target.squeeze(1).float()
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = F.one_hot(pred, self.num_classes).permute(0, 3, 1, 2).float()
            dice = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice += (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice /= self.num_classes
        return dice


class IoU(nn.Module):

    def __init__(self, num_classes=1) ->None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.0

    def forward(self, pred, target):
        target = target.squeeze(1).float()
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            den = union - intersection
            iou = (intersection + self.smooth) / (den + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = F.one_hot(pred, self.num_classes).permute(0, 3, 1, 2).float()
            iou = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                den = union - intersection
                iou += (intersection + self.smooth) / (den + self.smooth)
            iou /= self.num_classes
        return iou


class local_conv_block(nn.Module):

    def __init__(self, in_features, out_features, norm=True, activation=True) ->None:
        super().__init__()
        self.norm = norm
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(5, 5), padding=(2, 2))
        if self.norm:
            self.bn = nn.BatchNorm2d(out_features)
        if self.activation:
            self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.prelu(x)
        return x


class ScaleDotProduct(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        x12 = torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.einsum('bhcw, bhwk -> bhck', att, x3)
        return x123


class depthwise_conv_block(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=None, norm_type='bn', activation=True, use_bias=True, pointwise=False):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(in_channels=in_features, out_channels=in_features if pointwise else out_features, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features, out_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=use_bias)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class depthwise_projection(nn.Module):

    def __init__(self, in_features, out_features, groups, kernel_size=(1, 1), padding=(0, 0), norm_type=None, activation=False, pointwise=False) ->None:
        super().__init__()
        self.proj = depthwise_conv_block(in_features=in_features, out_features=out_features, kernel_size=kernel_size, padding=padding, groups=groups, pointwise=pointwise, norm_type=norm_type, activation=activation)

    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P)
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x


class ChannelAttention(nn.Module):

    def __init__(self, in_features, out_features, n_heads=1) ->None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features, out_features=out_features, groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features, out_features=in_features, groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features, out_features=in_features, groups=in_features)
        self.projection = depthwise_projection(in_features=out_features, out_features=out_features, groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att


class SpatialAttention(nn.Module):

    def __init__(self, in_features, out_features, n_heads=4) ->None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=in_features, out_features=in_features, groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features, out_features=in_features, groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features, out_features=out_features, groups=out_features)
        self.projection = depthwise_projection(in_features=out_features, out_features=out_features, groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        x = self.projection(att)
        return x


class CCSABlock(nn.Module):

    def __init__(self, features, channel_head, spatial_head, spatial_att=True, channel_att=True) ->None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-06) for in_features in features])
            self.c_attention = nn.ModuleList([ChannelAttention(in_features=sum(features), out_features=feature, n_heads=head) for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-06) for in_features in features])
            self.s_attention = nn.ModuleList([SpatialAttention(in_features=sum(features), out_features=feature, n_heads=head) for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [(xi + xj) for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat(args, dim=2)


class PoolEmbedding(nn.Module):

    def __init__(self, pooling, patch) ->None:
        super().__init__()
        self.projection = pooling(output_size=(patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x


class conv_block(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), norm_type='bn', activation=True, use_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=use_bias)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class UpsampleConv(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3), padding=(1, 1), norm_type=None, activation=False, scale=(2, 2), conv='conv') ->None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        if conv == 'conv':
            self.conv = conv_block(in_features=in_features, out_features=out_features, kernel_size=(1, 1), padding=(0, 0), norm_type=norm_type, activation=activation)
        elif conv == 'depthwise':
            self.conv = depthwise_conv_block(in_features=in_features, out_features=out_features, kernel_size=kernel_size, padding=padding, norm_type=norm_type, activation=activation)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class DCA(nn.Module):

    def __init__(self, features, strides, patch=28, channel_att=True, spatial_att=True, n=1, channel_head=[1, 1, 1, 1], spatial_head=[4, 4, 4, 4]):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(pooling=nn.AdaptiveAvgPool2d, patch=patch) for _ in features])
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature, out_features=feature, kernel_size=(1, 1), padding=(0, 0), groups=feature) for feature in features])
        self.attention = nn.ModuleList([CCSABlock(features=features, channel_head=channel_head, spatial_head=spatial_head, channel_att=channel_att, spatial_att=spatial_att) for _ in range(n)])
        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature, out_features=feature, kernel_size=(1, 1), padding=(0, 0), norm_type=None, activation=False, scale=stride, conv='conv') for feature, stride in zip(features, strides)])
        self.bn_relu = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(feature), nn.ReLU()) for feature in features])

    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg)
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return *x_out,

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [(xi + xj) for xi, xj in zip(x, y)]

    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch)


class DoubleASPP(nn.Module):

    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()
        self.block1 = conv_block(in_features=in_features, out_features=out_features, kernel_size=(1, 1), padding=(0, 0), norm_type=norm_type, activation=activation)
        self.block2 = conv_block(in_features=in_features, out_features=out_features, padding=rate[0], dilation=rate[0], norm_type=norm_type, activation=activation, use_bias=False)
        self.block3 = conv_block(in_features=in_features, out_features=out_features, padding=rate[1], dilation=rate[1], norm_type=norm_type, activation=activation, use_bias=False)
        self.block4 = conv_block(in_features=in_features, out_features=out_features, padding=rate[2], dilation=rate[2], norm_type=norm_type, activation=activation, use_bias=False)
        self.block5 = conv_block(in_features=in_features, out_features=out_features, padding=rate[3], dilation=rate[3], norm_type=norm_type, activation=activation, use_bias=False)
        self.out = conv_block(in_features=out_features * 5, out_features=out_features, kernel_size=(1, 1), padding=(0, 0), norm_type=norm_type, activation=activation, use_bias=False)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x5 = self.block5(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.out(x)
        return x


class DoubleUnet(nn.Module):

    def __init__(self, attention=False, n=1, in_features=3, out_features=3, k=1, input_size=(512, 512), patch_size=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4, 4], channel_head_dim=[1, 1, 1, 1], device='cuda') ->None:
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.attention = attention
        patch = input_size[0] // patch_size
        pretrained = False
        self.mu = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view((1, 3, 1, 1))
        self.sigma = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view((1, 3, 1, 1))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.vgg1 = vgg19(pretrained=pretrained).features[:3]
        self.vgg2 = vgg19(pretrained=pretrained).features[4:8]
        self.vgg3 = vgg19(pretrained=pretrained).features[9:17]
        self.vgg4 = vgg19(pretrained=pretrained).features[18:26]
        self.vgg5 = vgg19(pretrained=pretrained).features[27:-2]
        for m in [self.vgg1, self.vgg2, self.vgg3, self.vgg4, self.vgg5]:
            for param in m.parameters():
                param.requires_grad = True
        self.aspp_1 = DoubleASPP(in_features=512, out_features=64)
        if self.attention:
            self.DCA_vgg1 = DCA(n=n, features=[64, 128, 256, 512], strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8], patch=patch, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim)
            self.DCA_vgg2 = DCA(n=n, features=[64, 128, 256, 512], strides=[patch_size_ratio, patch_size_ratio // 2, patch_size_ratio // 4, patch_size_ratio // 8], patch_size=patch_size, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim)
            self.DCA = DCA(n=n, features=[32, 64, 128, 256], strides=[patch_size_ratio, patch_size_ratio // 2, patch_size_ratio // 4, patch_size_ratio // 8], patch_size=patch_size, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim)
        self.decode1 = local_conv_block(in_features=64 + 512, out_features=256)
        self.decode2 = local_conv_block(in_features=256 + 256, out_features=128)
        self.decode3 = local_conv_block(in_features=128 + 128, out_features=64)
        self.decode4 = local_conv_block(in_features=64 + 64, out_features=32)
        self.out1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=in_features, kernel_size=(1, 1), padding=(0, 0)), nn.Sigmoid())
        self.encode2_1 = local_conv_block(in_features=in_features, out_features=32)
        self.encode2_2 = local_conv_block(in_features=32, out_features=64)
        self.encode2_3 = local_conv_block(in_features=64, out_features=128)
        self.encode2_4 = local_conv_block(in_features=128, out_features=256)
        self.aspp_2 = DoubleASPP(in_features=256, out_features=64)
        self.decode2_1 = local_conv_block(in_features=64 + 512 + 256, out_features=256)
        self.decode2_2 = local_conv_block(in_features=256 + 256 + 128, out_features=128)
        self.decode2_3 = local_conv_block(in_features=128 + 128 + 64, out_features=64)
        self.decode2_4 = local_conv_block(in_features=64 + 64 + 32, out_features=32)
        self.out = nn.Conv2d(in_channels=32, out_channels=out_features, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x_in):
        if x_in.shape[1] == 1:
            x_in = torch.cat((x_in, x_in, x_in), dim=1)
        x_in = self.normalize(x_in)
        x1 = self.vgg1(x_in)
        x = self.relu(x1)
        x2 = self.vgg2(x)
        x = self.relu(x2)
        x3 = self.vgg3(x)
        x = self.relu(x3)
        x4 = self.vgg4(x)
        x = self.relu(x4)
        x = self.vgg5(x)
        x = self.relu(x)
        x = self.aspp_1(x)
        if self.attention:
            x1, x2, x3, x4 = self.DCA_vgg1([x1, x2, x3, x4])
            x12, x22, x32, x42 = self.DCA_vgg2([x1, x2, x3, x4])
        x = self.upsample(x)
        x = torch.cat((x4, x), dim=1)
        x, _ = self.decode1(x)
        x = self.upsample(x)
        x = torch.cat((x3, x), dim=1)
        x, _ = self.decode2(x)
        x = self.upsample(x)
        x = torch.cat((x2, x), dim=1)
        x, _ = self.decode3(x)
        x = self.upsample(x)
        x = torch.cat((x1, x), dim=1)
        x, _ = self.decode4(x)
        x = self.out1(x)
        out = x * x_in
        x, x1_2 = self.encode2_1(out)
        x = self.maxpool(x)
        x, x2_2 = self.encode2_2(x)
        x = self.maxpool(x)
        x, x3_2 = self.encode2_3(x)
        x = self.maxpool(x)
        x, x4_2 = self.encode2_4(x)
        x = self.maxpool(x)
        x = self.aspp_2(x)
        if self.attention:
            x1_2, x2_2, x3_2, x4_2 = self.DCA([x1_2, x2_2, x3_2, x4_2])
        x = self.upsample(x)
        x = torch.cat((x42, x4_2, x), dim=1)
        x, _ = self.decode2_1(x)
        x = self.upsample(x)
        x = torch.cat((x32, x3_2, x), dim=1)
        x, _ = self.decode2_2(x)
        x = self.upsample(x)
        x = torch.cat((x22, x2_2, x), dim=1)
        x, _ = self.decode2_3(x)
        x = self.upsample(x)
        x = torch.cat((x12, x1_2, x), dim=1)
        x, _ = self.decode2_4(x)
        x = self.out(x)
        return x

    def normalize(self, x):
        return (x - self.mu) / self.sigma


class MultiResBlock(nn.Module):

    def __init__(self, in_features, filters) ->None:
        super().__init__()
        f1 = int(1.67 * filters * 0.167)
        f2 = int(1.67 * filters * 0.333)
        f3 = int(1.67 * filters * 0.5)
        fout = f1 + f2 + f3
        self.skip = conv_block(in_features=in_features, out_features=fout, kernel_size=(1, 1), padding=(0, 0), norm_type='bn', activation=False)
        self.c1 = conv_block(in_features=in_features, out_features=f1, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)
        self.c2 = conv_block(in_features=f1, out_features=f2, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)
        self.c3 = conv_block(in_features=f2, out_features=f3, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)
        self.bn1 = nn.BatchNorm2d(fout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.bn1(x)
        x += x_skip
        x = self.relu(x)
        return x


class ResPath(nn.Module):

    def __init__(self, in_features, out_features, n) ->None:
        super().__init__()
        self.n = n
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_features) for _ in range(n)])
        self.skips = nn.ModuleList([conv_block(in_features=in_features, out_features=out_features, kernel_size=(1, 1), padding=(0, 0), norm_type=None, activation=False)])
        self.convs = nn.ModuleList([conv_block(in_features=in_features, out_features=out_features, kernel_size=(3, 3), padding=(1, 1), norm_type=None, activation=False)])
        for _ in range(n - 1):
            self.skips.append(conv_block(in_features=out_features, out_features=out_features, kernel_size=(1, 1), padding=(0, 0), norm_type=None, activation=False))
            self.convs.append(conv_block(in_features=out_features, out_features=out_features, kernel_size=(3, 3), padding=(1, 1), norm_type=None, activation=False))
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.n):
            x_skip = self.skips[i](x)
            x = self.convs[i](x)
            x_s = x + x_skip
            x = self.bns[i](x_s)
            x = self.relu(x)
        return x, x_s


class MultiResUnet(nn.Module):

    def __init__(self, attention=False, n=1, in_features=3, out_features=3, k=1, input_size=(512, 512), patch_size=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4, 4], channel_head_dim=[1, 1, 1, 1], device='cuda') ->None:
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.attention = attention
        patch = input_size[0] // patch_size
        alpha = 1.67
        k = 1
        in_filters1 = int(32 * alpha * 0.167) + int(32 * alpha * 0.333) + int(32 * alpha * 0.5)
        in_filters2 = int(32 * 2 * alpha * 0.167) + int(32 * 2 * alpha * 0.333) + int(32 * 2 * alpha * 0.5)
        in_filters3 = int(32 * 4 * alpha * 0.167) + int(32 * 4 * alpha * 0.333) + int(32 * 4 * alpha * 0.5)
        in_filters4 = int(32 * 8 * alpha * 0.167) + int(32 * 8 * alpha * 0.333) + int(32 * 8 * alpha * 0.5)
        in_filters5 = int(32 * 16 * alpha * 0.167) + int(32 * 16 * alpha * 0.333) + int(32 * 16 * alpha * 0.5)
        in_filters6 = int(32 * 8 * alpha * 0.167) + int(32 * 8 * alpha * 0.333) + int(32 * 8 * alpha * 0.5)
        in_filters7 = int(32 * 4 * alpha * 0.167) + int(32 * 4 * alpha * 0.333) + int(32 * 4 * alpha * 0.5)
        in_filters8 = int(32 * 2 * alpha * 0.167) + int(32 * 2 * alpha * 0.333) + int(32 * 2 * alpha * 0.5)
        in_filters9 = int(32 * alpha * 0.167) + int(32 * alpha * 0.333) + int(32 * alpha * 0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.mr1 = MultiResBlock(in_features=in_features, filters=int(32 * k))
        self.respath1 = ResPath(in_features=in_filters1, out_features=32, n=4)
        self.mr2 = MultiResBlock(in_features=in_filters1, filters=int(32 * k * 2))
        self.respath2 = ResPath(in_features=in_filters2, out_features=32 * 2, n=3)
        self.mr3 = MultiResBlock(in_features=in_filters2, filters=int(32 * k * 4))
        self.respath3 = ResPath(in_features=in_filters3, out_features=32 * 4, n=2)
        self.mr4 = MultiResBlock(in_features=in_filters3, filters=int(32 * k * 8))
        self.respath4 = ResPath(in_features=in_filters4, out_features=32 * 8, n=1)
        self.mr5 = MultiResBlock(in_features=in_filters4, filters=int(32 * k * 16))
        if self.attention:
            self.DCA = DCA(n=n, features=[int(32 * k), int(64 * k), int(128 * k), int(256 * k)], strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8], patch=patch, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_filters5, 32 * 8, kernel_size=(2, 2), stride=(2, 2)), nn.BatchNorm2d(32 * 8), nn.ReLU())
        self.mr6 = MultiResBlock(in_features=32 * 8 * 2, filters=int(32 * k * 8))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(in_filters6, 32 * 4, kernel_size=(2, 2), stride=(2, 2)), nn.BatchNorm2d(32 * 4), nn.ReLU())
        self.mr7 = MultiResBlock(in_features=32 * 4 * 2, filters=int(32 * k * 4))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(in_filters7, 32 * 2, kernel_size=(2, 2), stride=(2, 2)), nn.BatchNorm2d(32 * 2), nn.ReLU())
        self.mr8 = MultiResBlock(in_features=32 * 2 * 2, filters=int(32 * k * 2))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(in_filters8, 32, kernel_size=(2, 2), stride=(2, 2)), nn.BatchNorm2d(32), nn.ReLU())
        self.mr9 = MultiResBlock(in_features=32 * 2, filters=int(32 * k))
        self.out = nn.Conv2d(in_channels=in_filters9, out_channels=out_features, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        x1 = self.mr1(x)
        xp1 = self.maxpool(x1)
        x1, x1_ = self.respath1(x1)
        x2 = self.mr2(xp1)
        xp2 = self.maxpool(x2)
        x2, x2_ = self.respath2(x2)
        x3 = self.mr3(xp2)
        xp3 = self.maxpool(x3)
        x3, x3_ = self.respath3(x3)
        x4 = self.mr4(xp3)
        xp4 = self.maxpool(x4)
        x4, x4_ = self.respath4(x4)
        x = self.mr5(xp4)
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1_, x2_, x3_, x4_])
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.mr6(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.mr7(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.mr8(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.mr9(x)
        x = self.out(x)
        return x


class Upconv(nn.Module):

    def __init__(self, in_features, out_features, activation=True, norm_type='bn', scale=(2, 2)) ->None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.conv = conv_block(in_features=in_features, out_features=out_features, norm_type=norm_type, activation=activation)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class rec_block(nn.Module):

    def __init__(self, in_features, out_features, norm_type='bn', activation=True, t=2):
        super().__init__()
        self.t = t
        self.conv = conv_block(in_features=in_features, out_features=out_features, norm_type=norm_type, activation=activation)

    def forward(self, x):
        x1 = self.conv(x)
        for _ in range(self.t):
            x1 = self.conv(x + x1)
        return x1


class rrcnn_block(nn.Module):

    def __init__(self, in_features, out_features, norm_type='bn', activation=True, t=2):
        super().__init__()
        self.conv = conv_block(in_features=in_features, out_features=out_features, kernel_size=(1, 1), padding=(0, 0), norm_type=None, activation=False)
        self.block = nn.Sequential(rec_block(in_features=out_features, out_features=out_features, t=t, norm_type=norm_type, activation=activation), rec_block(in_features=out_features, out_features=out_features, t=t, norm_type=None, activation=False))
        self.norm = nn.BatchNorm2d(out_features)
        self.norm_c = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x1 = self.norm_c(x)
        x1 = self.relu(x1)
        x1 = self.block(x1)
        xs = x + x1
        x = self.norm(xs)
        x = self.relu(x)
        return x, xs


class R2Unet(nn.Module):

    def __init__(self, attention=False, n=1, in_features=3, out_features=3, k=0.5, input_size=(512, 512), patch_size=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4, 4], channel_head_dim=[1, 1, 1, 1], device='cuda') ->None:
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.attention = attention
        patch = input_size[0] // patch_size
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.rconv1 = rrcnn_block(in_features=in_features, out_features=int(64 * k))
        self.rconv2 = rrcnn_block(in_features=int(64 * k), out_features=int(128 * k))
        self.rconv3 = rrcnn_block(in_features=int(128 * k), out_features=int(256 * k))
        self.rconv4 = rrcnn_block(in_features=int(256 * k), out_features=int(512 * k))
        self.rconv5 = rrcnn_block(in_features=int(512 * k), out_features=int(1024 * k))
        if self.attention:
            self.DCA = DCA(n=n, features=[int(64 * k), int(128 * k), int(256 * k), int(512 * k)], strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8], patch=patch, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim)
        self.up1 = Upconv(in_features=int(1024 * k), out_features=int(512 * k))
        self.rconv6 = rrcnn_block(in_features=int(1024 * k), out_features=int(512 * k))
        self.up2 = Upconv(in_features=int(512 * k), out_features=int(256 * k))
        self.rconv7 = rrcnn_block(in_features=int(512 * k), out_features=int(256 * k))
        self.up3 = Upconv(in_features=int(256 * k), out_features=int(128 * k))
        self.rconv8 = rrcnn_block(in_features=int(256 * k), out_features=int(128 * k))
        self.up4 = Upconv(in_features=int(128 * k), out_features=int(64 * k))
        self.rconv9 = rrcnn_block(in_features=int(128 * k), out_features=int(64 * k))
        self.out = nn.Conv2d(in_channels=int(64 * k), out_channels=out_features, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        x1, x1_ = self.rconv1(x)
        x2 = self.maxpool(x1)
        x2, x2_ = self.rconv2(x2)
        x3 = self.maxpool(x2)
        x3, x3_ = self.rconv3(x3)
        x4 = self.maxpool(x3)
        x4, x4_ = self.rconv4(x4)
        x = self.maxpool(x4)
        x, _ = self.rconv5(x)
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1_, x2_, x3_, x4_])
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x, _ = self.rconv6(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x, _ = self.rconv7(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x, _ = self.rconv8(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x, _ = self.rconv9(x)
        x = self.out(x)
        return x


class ASPP(nn.Module):

    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()
        self.block1 = conv_block(in_features=in_features, out_features=out_features, padding=rate[0], dilation=rate[0], norm_type=norm_type, activation=activation)
        self.block2 = conv_block(in_features=in_features, out_features=out_features, padding=rate[1], dilation=rate[1], norm_type=norm_type, activation=activation)
        self.block3 = conv_block(in_features=in_features, out_features=out_features, padding=rate[2], dilation=rate[2], norm_type=norm_type, activation=activation)
        self.block4 = conv_block(in_features=in_features, out_features=out_features, padding=rate[3], dilation=rate[3], norm_type=norm_type, activation=activation)
        self.out = conv_block(in_features=out_features, out_features=out_features, kernel_size=(1, 1), padding=(0, 0), norm_type=norm_type, activation=activation)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x = x1 + x2 + x3 + x4
        x = self.out(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, input_encoder, input_decoder, output_dim, norm_type='bn'):
        super().__init__()
        if norm_type == 'gn':
            self.norm1 = nn.GroupNorm(32 if input_encoder >= 32 and input_encoder % 32 == 0 else input_encoder, input_encoder)
            self.norm2 = nn.GroupNorm(32 if input_decoder >= 32 and input_decoder % 32 == 0 else input_decoder, input_decoder)
            self.norm3 = nn.GroupNorm(32 if output_dim >= 32 and output_dim % 32 == 0 else output_dim, output_dim)
        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(input_encoder)
            self.norm2 = nn.BatchNorm2d(input_decoder)
            self.norm3 = nn.BatchNorm2d(output_dim)
        else:
            self.norm1, self.norm2, self.norm3 = nn.Identity(), nn.Identity(), nn.Identity()
        self.conv_encoder = nn.Sequential(self.norm1, nn.ReLU(), nn.Conv2d(input_encoder, output_dim, 3, padding=1), nn.MaxPool2d(2, 2))
        self.conv_decoder = nn.Sequential(self.norm2, nn.ReLU(), nn.Conv2d(input_decoder, output_dim, 3, padding=1))
        self.conv_attn = nn.Sequential(self.norm3, nn.ReLU(), nn.Conv2d(output_dim, 1, 1))

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class bn_relu(nn.Module):

    def __init__(self, features) ->None:
        super().__init__()
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(x))


class ResConv(nn.Module):

    def __init__(self, in_features, out_features, stride=(1, 1)):
        super().__init__()
        self.conv = nn.Sequential(bn_relu(in_features), nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(3, 3), padding=(1, 1), stride=stride), bn_relu(out_features), nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)))
        self.skip = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1), padding=(0, 0), stride=stride)

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class SqueezeExciteBlock(nn.Module):

    def __init__(self, in_features, reduction: 'int'=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=False), nn.ReLU(), nn.Linear(int(in_features // reduction), in_features, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out


class ResUnetPlus(nn.Module):

    def __init__(self, attention=False, n=1, in_features=3, out_features=3, k=0.5, input_size=(512, 512), fusion_out=None, patch_size=4, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4], channel_head_dim=[1, 1, 1], device='cuda') ->None:
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.attention = attention
        patch = input_size[0] // patch_size
        self.input_layer = nn.Sequential(conv_block(in_features=in_features, out_features=int(64 * k)), nn.Conv2d(int(64 * k), int(64 * k), kernel_size=3, padding=1))
        self.input_skip = nn.Sequential(nn.Conv2d(in_features, int(64 * k), kernel_size=3, padding=1))
        self.squeeze_excite1 = SqueezeExciteBlock(int(64 * k), reduction=int(16 * k))
        self.residual_conv1 = ResConv(int(64 * k), int(128 * k), stride=2)
        self.squeeze_excite2 = SqueezeExciteBlock(int(128 * k), reduction=int(32 * k))
        self.residual_conv2 = ResConv(int(128 * k), int(256 * k), stride=2)
        self.squeeze_excite3 = SqueezeExciteBlock(int(256 * k), reduction=int(32 * k))
        self.residual_conv3 = ResConv(int(256 * k), int(512 * k), stride=2)
        self.aspp_bridge = ASPP(int(512 * k), int(1024 * k), norm_type='bn', activation=False)
        if self.attention:
            self.DCA = DCA(n=n, features=[int(64 * k), int(128 * k), int(256 * k)], strides=[patch_size, patch_size // 2, patch_size // 4], patch=patch, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim)
        self.attn1 = AttentionBlock(int(256 * k), int(1024 * k), int(1024 * k))
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_residual_conv1 = ResConv(int(1024 * k) + int(256 * k), int(512 * k))
        self.attn2 = AttentionBlock(int(128 * k), int(512 * k), int(512 * k))
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_residual_conv2 = ResConv(int(512 * k) + int(128 * k), int(256 * k))
        self.attn3 = AttentionBlock(int(64 * k), int(256 * k), int(256 * k))
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_residual_conv3 = ResConv(int(256 * k) + int(64 * k), int(128 * k))
        self.aspp_out = ASPP(int(128 * k), int(64 * k))
        self.output_layer = nn.Conv2d(int(64 * k), out_features, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)
        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)
        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)
        x5 = self.aspp_bridge(x4)
        if self.attention:
            x1, x2, x3 = self.DCA([x1, x2, x3])
        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)
        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)
        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)
        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        return out


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def to_2tuple(x):
    return x, x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) ->str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale ** 2)
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerSys(nn.Module):

    def __init__(self, attention_type='dca', attention=False, n=1, k=0.5, patch_size_ratio=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4], channel_head_dim=[1, 1, 1], img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, final_upsample='expand_first', **kwargs):
        super().__init__()
        None
        self.attention = attention
        patch_size_d = img_size // (patch_size_ratio * 2)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        if self.attention:
            self.DCA = DCA(n=n, features=[96, 192, 384], strides=[patch_size_ratio // 2, patch_size_ratio // 4, patch_size_ratio // 8], patch_size=patch_size_d, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim, attention_type=attention_type)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // 2 ** (self.num_layers - 1 - i_layer), patches_resolution[1] // 2 ** (self.num_layers - 1 - i_layer)), dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), input_resolution=(patches_resolution[0] // 2 ** (self.num_layers - 1 - i_layer), patches_resolution[1] // 2 ** (self.num_layers - 1 - i_layer)), depth=depths[self.num_layers - 1 - i_layer], num_heads=num_heads[self.num_layers - 1 - i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:self.num_layers - 1 - i_layer]):sum(depths[:self.num_layers - 1 - i_layer + 1])], norm_layer=norm_layer, upsample=PatchExpand if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        if self.final_upsample == 'expand_first':
            None
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size), dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, 'input features has wrong size'
        if self.final_upsample == 'expand_first':
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.output(x)
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        if self.attention:
            last = x_downsample[-1]
            x_downsample.pop()
            x_downsample = [self.b_c_h_w(i) for i in x_downsample]
            x1, x2, x3 = self.DCA(x_downsample)
            x_downsample = [x1, x2, x3]
            x_downsample = [self.b_hw_c(i) for i in x_downsample]
            x_downsample += [last]
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x

    def b_c_h_w(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=int(x.shape[1] ** 0.5))

    def b_hw_c(self, x):
        return einops.rearrange(x, 'B C H W -> B (H W) C')

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class SwinUnet(nn.Module):

    def __init__(self, attention_type='dca', attention=False, n=1, in_features=3, out_features=1, k=0.5, input_size=(224, 224), patch_size_ratio=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4], channel_head_dim=[1, 1, 1], vis=False, device='cuda'):
        super(SwinUnet, self).__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.num_classes = out_features
        self.zero_head = False
        self.swin_unet = SwinTransformerSys(attention_type=attention_type, attention=attention, n=n, k=k, patch_size_ratio=patch_size_ratio, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4], channel_head_dim=[1, 1, 1], img_size=input_size[0], patch_size=4, in_chans=in_features, num_classes=self.num_classes, embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_path_rate=0.2)
        self.load_from()

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self):
        pretrained_path = 'model/checkpoints/swin_unet.pth'
        if pretrained_path is not None:
            None
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            full_dict = pretrained_dict
            del full_dict['output.weight']
            self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            None


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, head_num):
        super().__init__()
        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)
        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)
        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum('... i d , ... j d -> ... i j', query, key) * self.dk
        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)
        attention = torch.softmax(energy, dim=-1)
        x = torch.einsum('... i j , ... j d -> ... i d', attention, value)
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.out_attention(x)
        return x


class MLP(nn.Module):

    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.mlp_layers = nn.Sequential(nn.Linear(embedding_dim, mlp_dim), GELU(), nn.Dropout(0.1), nn.Linear(mlp_dim, embedding_dim), nn.Dropout(0.1))

    def forward(self, x):
        x = self.mlp_layers(x)
        return x


class TransformerEncoderBlock(nn.Module):

    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)
        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        self.layer_blocks = nn.ModuleList([TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)
        return x


class ViT(nn.Module):

    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()
        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * patch_dim ** 2
        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)
        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)', patch_x=self.patch_dim, patch_y=self.patch_dim)
        batch_size, tokens, _ = img_patches.shape
        project = self.projection(img_patches)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)
        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :]
        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]
        return x


class EncoderBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        width = int(out_channels * (base_width / 64))
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)
        return x


class DecoderBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, x_concat=None):
        x = self.upsample(x)
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
        x = self.layer(x)
        return x


class Encoder(nn.Module):

    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)
        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8, head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = self.vit(x)
        x = rearrange(x, 'b (x y) c -> b c x y', x=self.vit_img_dim, y=self.vit_img_dim)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x, x1, x2, x3


class Decoder(nn.Module):

    def __init__(self, out_channels, class_num):
        super().__init__()
        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))
        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        return x


class TransUNet(nn.Module):

    def __init__(self, attention_type='dca', attention=False, n=1, in_features=3, out_features=3, k=0.5, input_size=(224, 224), patch_size_ratio=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4], channel_head_dim=[1, 1, 1], out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, device='cuda'):
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.attention = attention
        patch_size = input_size[0] // patch_size_ratio
        self.encoder = Encoder(input_size[0], in_features, out_channels, head_num, mlp_dim, block_num, patch_dim)
        if self.attention:
            self.DCA = DCA(n=n, features=[128, 256, 512], strides=[patch_size_ratio // 2, patch_size_ratio // 4, patch_size_ratio // 8], patch_size=patch_size, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim, attention_type=attention_type)
        self.decoder = Decoder(out_channels, out_features)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x1, x2, x3 = self.DCA([x1, x2, x3])
        x = self.decoder(x, x1, x2, x3)
        return x


class double_conv_block(nn.Module):

    def __init__(self, in_features, out_features1, out_features2, *args, **kwargs):
        super().__init__()
        self.conv1 = conv_block(*args, in_features=in_features, out_features=out_features1, **kwargs)
        self.conv2 = conv_block(*args, in_features=out_features1, out_features=out_features2, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class double_conv_block_a(nn.Module):

    def __init__(self, in_features, out_features1, out_features2, norm1, norm2, act1, act2, *args, **kwargs):
        super().__init__()
        self.conv1 = conv_block(*args, in_features=in_features, out_features=out_features1, norm_type=norm1, activation=act1, **kwargs)
        self.conv2 = conv_block(*args, in_features=out_features1, out_features=out_features2, norm_type=norm2, activation=act2, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Unet(nn.Module):

    def __init__(self, attention=False, n=1, in_features=3, out_features=3, k=0.5, input_size=(512, 512), patch_size=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4, 4], channel_head_dim=[1, 1, 1, 1], device='cuda') ->None:
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.attention = attention
        patch = input_size[0] // patch_size
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        norm2 = None
        self.conv1 = double_conv_block_a(in_features=in_features, out_features1=int(64 * k), out_features2=int(64 * k), norm1='bn', norm2=norm2, act1=True, act2=False)
        self.norm1 = nn.BatchNorm2d(int(64 * k))
        self.conv2 = double_conv_block_a(in_features=int(64 * k), out_features1=int(128 * k), out_features2=int(128 * k), norm1='bn', norm2=norm2, act1=True, act2=False)
        self.norm2 = nn.BatchNorm2d(int(128 * k))
        self.conv3 = double_conv_block_a(in_features=int(128 * k), out_features1=int(256 * k), out_features2=int(256 * k), norm1='bn', norm2=norm2, act1=True, act2=False)
        self.norm3 = nn.BatchNorm2d(int(256 * k))
        self.conv4 = double_conv_block_a(in_features=int(256 * k), out_features1=int(512 * k), out_features2=int(512 * k), norm1='bn', norm2=norm2, act1=True, act2=False)
        self.norm4 = nn.BatchNorm2d(int(512 * k))
        self.conv5 = double_conv_block(in_features=int(512 * k), out_features1=int(1024 * k), out_features2=int(1024 * k), norm_type='bn')
        if self.attention:
            self.DCA = DCA(n=n, features=[int(64 * k), int(128 * k), int(256 * k), int(512 * k)], strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8], patch=patch, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim)
        self.up1 = Upconv(in_features=int(1024 * k), out_features=int(512 * k), norm_type='bn')
        self.upconv1 = double_conv_block(in_features=int(512 * k + 512 * k), out_features1=int(512 * k), out_features2=int(512 * k), norm_type='bn')
        self.up2 = Upconv(in_features=int(512 * k), out_features=int(256 * k), norm_type='bn')
        self.upconv2 = double_conv_block(in_features=int(256 * k + 256 * k), out_features1=int(256 * k), out_features2=int(256 * k), norm_type='bn')
        self.up3 = Upconv(in_features=int(256 * k), out_features=int(128 * k), norm_type='bn')
        self.upconv3 = double_conv_block(in_features=int(128 * k + 128 * k), out_features1=int(128 * k), out_features2=int(128 * k), norm_type='bn')
        self.up4 = Upconv(in_features=int(128 * k), out_features=int(64 * k), norm_type='bn')
        self.upconv4 = double_conv_block(in_features=int(64 * k + 64 * k), out_features1=int(64 * k), out_features2=int(64 * k), norm_type='bn')
        self.out = conv_block(in_features=int(64 * k), out_features=out_features, norm_type=None, activation=False, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        x1 = self.conv1(x)
        x1_n = self.norm1(x1)
        x1_a = self.relu(x1_n)
        x2 = self.maxpool(x1_a)
        x2 = self.conv2(x2)
        x2_n = self.norm2(x2)
        x2_a = self.relu(x2_n)
        x3 = self.maxpool(x2_a)
        x3 = self.conv3(x3)
        x3_n = self.norm3(x3)
        x3_a = self.relu(x3_n)
        x4 = self.maxpool(x3_a)
        x4 = self.conv4(x4)
        x4_n = self.norm4(x4)
        x4_a = self.relu(x4_n)
        x5 = self.maxpool(x4_a)
        x = self.conv5(x5)
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1, x2, x3, x4])
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.upconv1(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.upconv2(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.upconv3(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.upconv4(x)
        x = self.out(x)
        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class shiftmlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.0, shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), 'constant', 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x = self.fc1(x_shift_r)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), 'constant', 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = img_size, img_size
        patch_size = patch_size, patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class UNext(nn.Module):

    def __init__(self, attention_type='dca', attention=False, n=1, in_features=3, out_features=1, k=0.5, input_size=(224, 224), patch_size_ratio=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4], channel_head_dim=[1, 1, 1], deep_supervision=False, patch_size=16, embed_dims=[128, 160, 256], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], device='cuda', **kwargs):
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.attention = attention
        patch_size = input_size[0] // patch_size_ratio
        if self.attention:
            self.DCA = DCA(n=n, features=[16, 32, 128, 160], strides=[patch_size_ratio // 2, patch_size_ratio // 4, patch_size_ratio // 8, patch_size_ratio // 16], patch_size=patch_size, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim, attention_type=attention_type)
        self.encoder1 = nn.Conv2d(in_features, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([shiftedBlock(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([shiftedBlock(dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock1 = nn.ModuleList([shiftedBlock(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock2 = nn.ModuleList([shiftedBlock(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.patch_embed3 = OverlapPatchEmbed(img_size=input_size[0] // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=input_size[0] // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        self.final = nn.Conv2d(16, out_features, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if self.attention:
            t1, t2, t3, t4 = self.DCA([t1, t2, t3, t4])
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.final(out)
        return out


class conv_projection(nn.Module):

    def __init__(self, in_features, out_features) ->None:
        super().__init__()
        self.proj = conv_block(in_features=in_features, out_features=out_features, kernel_size=(1, 1), padding=(0, 0), norm_type=None, activation=False)

    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P)
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x


class PatchEmbedding(nn.Module):

    def __init__(self, in_features, out_features, size, patch=28, proj='conv') ->None:
        super().__init__()
        self.proj = proj
        if self.proj == 'conv':
            self.projection = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=size // patch_size, stride=size // patch_size, padding=(0, 0))

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Layernorm(nn.Module):

    def __init__(self, features, eps=1e-06) ->None:
        super().__init__()
        self.norm = nn.LayerNorm(features, eps=eps)

    def forward(self, x):
        H = x.shape[2]
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        x = self.norm(x)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=H)
        return x


class double_depthwise_convblock(nn.Module):

    def __init__(self, in_features, out_features1, out_features2, kernels_per_layer=1, normalization=None, activation=None):
        super().__init__()
        if normalization is None:
            normalization = [True, True]
        if activation is None:
            activation = [True, True]
        self.block1 = depthwise_conv_block(in_features, out_features1, kernels_per_layer=kernels_per_layer, normalization=normalization[0], activation=activation[0])
        self.block2 = depthwise_conv_block(out_features1, out_features2, kernels_per_layer=kernels_per_layer, normalization=normalization[1], activation=activation[1])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class transpose_conv_block(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), out_padding=(1, 1), dilation=(1, 1), norm_type='bn', activation=True, use_bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding, dilation=dilation, bias=use_bias)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class InputConv(nn.Module):

    def __init__(self, in_features, out_features) ->None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(5, 5), padding=(2, 2))
        self.conv_skip = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1), padding=(0, 0))
        self.bn = nn.BatchNorm2d(out_features)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x_skip = self.conv_skip(x)
        x = self.conv(x)
        xs = x + x_skip
        x = self.bn(x)
        x = self.prelu(x)
        return x, xs


class DownConv(nn.Module):

    def __init__(self, in_features, out_features, n) ->None:
        super().__init__()
        self.n = n
        self.down = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(2, 2), stride=(2, 2))
        self.conv = nn.ModuleList([local_conv_block(in_features=out_features, out_features=out_features) for _ in range(n - 1)])
        self.conv.append(local_conv_block(in_features=out_features, out_features=out_features, norm=False, activation=False))
        self.bn = nn.BatchNorm2d(out_features)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x_d = self.down(x)
        x = x_d.clone()
        for i in range(self.n):
            x = self.conv[i](x)
        x_s = x + x_d
        x = self.bn(x_s)
        x = self.prelu(x)
        return x, x_s


class UpConv(nn.Module):

    def __init__(self, in_features, enc_features, out_features, n) ->None:
        super().__init__()
        self.n = n
        self.up = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=(2, 2), stride=(2, 2))
        self.bn_in = nn.BatchNorm2d(out_features)
        self.prelu_in = nn.PReLU()
        self.conv = nn.ModuleList([local_conv_block(in_features=out_features + enc_features, out_features=out_features)])
        for _ in range(n - 2):
            self.conv.append(local_conv_block(in_features=out_features, out_features=out_features))
        self.conv.append(local_conv_block(in_features=out_features, out_features=out_features, norm=False, activation=False))
        self.bn = nn.BatchNorm2d(out_features)
        self.prelu = nn.PReLU()

    def forward(self, x_e, x_d):
        x_d = self.up(x_d)
        x = self.bn_in(x_d)
        x = self.prelu_in(x)
        x = torch.cat((x_e, x), dim=1)
        for i in range(self.n):
            x = self.conv[i](x)
        x += x_d
        x = self.bn(x)
        x = self.prelu(x)
        return x


class OutputConv(nn.Module):

    def __init__(self, in_features, enc_features, out_features) ->None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=(2, 2), stride=(2, 2))
        self.bn_in = nn.BatchNorm2d(out_features)
        self.prelu_in = nn.PReLU()
        self.conv = nn.Conv2d(in_channels=out_features + enc_features, out_channels=out_features, kernel_size=(5, 5), padding=(2, 2))
        self.bn = nn.BatchNorm2d(out_features)
        self.prelu = nn.PReLU()

    def forward(self, x_e, x_d):
        x_d = self.up(x_d)
        x = self.bn_in(x_d)
        x = self.prelu_in(x)
        x = torch.cat((x_e, x), dim=1)
        x = self.conv(x)
        x += x_d
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Vnet(nn.Module):

    def __init__(self, attention=False, n=1, in_features=3, out_features=3, k=0.5, input_size=(512, 512), patch_size=8, spatial_att=True, channel_att=True, spatial_head_dim=[4, 4, 4, 4], channel_head_dim=[1, 1, 1, 1], device='cuda') ->None:
        super().__init__()
        k = 1
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.attention = attention
        patch = input_size[0] // patch_size
        self.conv1 = InputConv(in_features=in_features, out_features=int(32 * k))
        self.conv2 = DownConv(in_features=int(32 * k), out_features=int(64 * k), n=2)
        self.conv3 = DownConv(in_features=int(64 * k), out_features=int(128 * k), n=3)
        self.conv4 = DownConv(in_features=int(128 * k), out_features=int(256 * k), n=3)
        self.conv5 = DownConv(in_features=int(256 * k), out_features=int(512 * k), n=3)
        if self.attention:
            self.DCA = DCA(n=n, features=[int(32 * k), int(64 * k), int(128 * k), int(256 * k)], strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8], patch=patch, spatial_att=spatial_att, channel_att=channel_att, spatial_head=spatial_head_dim, channel_head=channel_head_dim)
        self.up1 = UpConv(in_features=int(512 * k), enc_features=int(256 * k), out_features=int(256 * k), n=3)
        self.up2 = UpConv(in_features=int(256 * k), enc_features=int(128 * k), out_features=int(128 * k), n=3)
        self.up3 = UpConv(in_features=int(128 * k), enc_features=int(64 * k), out_features=int(64 * k), n=2)
        self.up4 = OutputConv(in_features=int(64 * k), enc_features=int(32 * k), out_features=int(32 * k))
        self.out = nn.Conv2d(in_channels=int(32 * k), out_channels=out_features, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        x1, x1_ = self.conv1(x)
        x2, x2_ = self.conv2(x1)
        x3, x3_ = self.conv3(x2)
        x4, x4_ = self.conv4(x3)
        x, _ = self.conv5(x4)
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1_, x2_, x3_, x4_])
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.out(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AttentionBlock,
     lambda: ([], {'input_encoder': 4, 'input_decoder': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 2, 2]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiceScore,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DoubleASPP,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownConv,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EncoderBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     True),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InputConv,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IoU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'embedding_dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiResBlock,
     lambda: ([], {'in_features': 4, 'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (OverlapPatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResConv,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResPath,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaleDotProduct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeExciteBlock,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upconv,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsampleConv,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (bn_relu,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (conv_block,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (double_conv_block,
     lambda: ([], {'in_features': 4, 'out_features1': 4, 'out_features2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (local_conv_block,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (rec_block,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (rrcnn_block,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (transpose_conv_block,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_gorkemcanates_Dual_Cross_Attention(_paritybench_base):
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

    def test_028(self):
        self._check(*TESTCASES[28])

