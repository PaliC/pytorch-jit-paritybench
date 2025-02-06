import sys
_module = sys.modules[__name__]
del sys
ImagenetDataset = _module
data = _module
samplers = _module
CMT = _module
cmt = _module
Transformers = _module
test = _module
train = _module
utils = _module
augments = _module
calculate_acc = _module
optimizer_step = _module
precise_bn = _module

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


from math import e


import torch


import random


import numpy as np


from torch.utils.data.dataset import Dataset


from torchvision.transforms import transforms as imagenet_transforms


import torch.distributed as dist


import math


import torch.nn as nn


import torch.nn.functional as F


from random import choice


import warnings


import time


import torch.multiprocessing as mp


from torch.utils.data.dataloader import DataLoader


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.distributed import DistributedSampler


from torch.utils.tensorboard import SummaryWriter


from torch.nn.parallel import DistributedDataParallel as DataParallel


from torch.utils.data import DataLoader


from torch.cuda.amp import autocast as autocast


from torch import optim as optim


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import AdamW


import itertools


import logging


from typing import Iterable


from typing import Any


from torch.distributed import ReduceOp


from torch.distributed import all_reduce


def make_pairs(x):
    """make the int -> tuple 
    """
    return x if isinstance(x, tuple) else (x, x)


class ConvDW3x3(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super(ConvDW3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=make_pairs(kernel_size), padding=make_pairs(1), groups=dim)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvGeluBN(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride_size, padding=1):
        """build the conv3x3 + gelu + bn module
        """
        super(ConvGeluBN, self).__init__()
        self.kernel_size = make_pairs(kernel_size)
        self.stride_size = make_pairs(stride_size)
        self.padding_size = make_pairs(padding)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv3x3_gelu_bn = nn.Sequential(nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=self.kernel_size, stride=self.stride_size, padding=self.padding_size), nn.GELU(), nn.BatchNorm2d(self.out_channel))

    def forward(self, x):
        x = self.conv3x3_gelu_bn(x)
        return x


class InvertedResidualFeedForward(nn.Module):

    def __init__(self, dim, dim_ratio=4.0):
        super(InvertedResidualFeedForward, self).__init__()
        output_dim = int(dim_ratio * dim)
        self.conv1x1_gelu_bn = ConvGeluBN(in_channel=dim, out_channel=output_dim, kernel_size=1, stride_size=1, padding=0)
        self.conv3x3_dw = ConvDW3x3(dim=output_dim)
        self.act = nn.Sequential(nn.GELU(), nn.BatchNorm2d(output_dim))
        self.conv1x1_pw = nn.Sequential(nn.Conv2d(output_dim, dim, 1, 1, 0), nn.BatchNorm2d(dim))

    def forward(self, x):
        x = self.conv1x1_gelu_bn(x)
        out = x + self.act(self.conv3x3_dw(x))
        out = self.conv1x1_pw(out)
        return out


def generate_relative_distance(number_size):
    """return relative distance, (number_size**2, number_size**2, 2)
    """
    indices = torch.tensor(np.array([[x, y] for x in range(number_size) for y in range(number_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    distances = distances + number_size - 1
    return distances


class LightMutilHeadSelfAttention(nn.Module):
    """calculate the self attention with down sample the resolution for k, v, add the relative position bias before softmax
    Args:
        dim (int) : features map channels or dims 
        num_heads (int) : attention heads numbers
        relative_pos_embeeding (bool) : relative position embeeding 
        no_distance_pos_embeeding (bool): no_distance_pos_embeeding
        features_size (int) : features shape
        qkv_bias (bool) : if use the embeeding bias
        qk_scale (float) : qk scale if None use the default 
        attn_drop (float) : attention dropout rate
        proj_drop (float) : project linear dropout rate
        sr_ratio (float)  : k, v resolution downsample ratio
    Returns:
        x : LMSA attention result, the shape is (B, H, W, C) that is the same as inputs.
    """

    def __init__(self, dim, num_heads=8, features_size=56, relative_pos_embeeding=False, no_distance_pos_embeeding=False, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1.0):
        super(LightMutilHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_pos_embeeding = relative_pos_embeeding
        self.no_distance_pos_embeeding = no_distance_pos_embeeding
        self.features_size = features_size
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        if self.relative_pos_embeeding:
            self.relative_indices = generate_relative_distance(self.features_size)
            self.position_embeeding = nn.Parameter(torch.randn(2 * self.features_size - 1, 2 * self.features_size - 1))
        elif self.no_distance_pos_embeeding:
            self.position_embeeding = nn.Parameter(torch.randn(self.features_size ** 2, self.features_size ** 2))
        else:
            self.position_embeeding = None
        if self.position_embeeding is not None:
            trunc_normal_(self.position_embeeding, std=0.2)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_q = rearrange(x, 'B C H W -> B (H W) C')
        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_reduce_resolution = self.sr(x)
            x_kv = rearrange(x_reduce_resolution, 'B C H W -> B (H W) C ')
            x_kv = self.norm(x_kv)
        else:
            x_kv = rearrange(x, 'B C H W -> B (H W) C ')
        kv_emb = rearrange(self.kv(x_kv), 'B N (dim h l ) -> l B h N dim', h=self.num_heads, l=2)
        k, v = kv_emb[0], kv_emb[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        q_n, k_n = q.shape[1], k.shape[2]
        if self.relative_pos_embeeding:
            attn = attn + self.position_embeeding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]][:, :k_n]
        elif self.no_distance_pos_embeeding:
            attn = attn + self.position_embeeding[:, :k_n]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'B (H W) C -> B C H W ', H=H, W=W)
        return x


class LocalPerceptionUint(nn.Module):

    def __init__(self, dim, act=False):
        super(LocalPerceptionUint, self).__init__()
        self.act = act
        self.conv_3x3_dw = ConvDW3x3(dim)
        if self.act:
            self.actation = nn.Sequential(nn.GELU(), nn.BatchNorm2d(dim))

    def forward(self, x):
        if self.act:
            out = self.actation(self.conv_3x3_dw(x))
            return out
        else:
            out = self.conv_3x3_dw(x)
            return out


class CMTLayers(nn.Module):

    def __init__(self, dim, num_heads=8, ffn_ratio=4.0, relative_pos_embeeding=True, no_distance_pos_embeeding=False, features_size=56, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1.0, drop_path_rate=0.0):
        super(CMTLayers, self).__init__()
        self.dim = dim
        self.ffn_ratio = ffn_ratio
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.LPU = LocalPerceptionUint(self.dim)
        self.LMHSA = LightMutilHeadSelfAttention(dim=self.dim, num_heads=num_heads, relative_pos_embeeding=relative_pos_embeeding, no_distance_pos_embeeding=no_distance_pos_embeeding, features_size=features_size, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.IRFFN = InvertedResidualFeedForward(self.dim, self.ffn_ratio)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        lpu = self.LPU(x)
        x = x + lpu
        b, c, h, w = x.shape
        x_1 = rearrange(x, 'b c h w -> b ( h w ) c ')
        norm1 = self.norm1(x_1)
        norm1 = rearrange(norm1, 'b ( h w ) c -> b c h w', h=h, w=w)
        attn = self.LMHSA(norm1)
        x = x + attn
        b, c, h, w = x.shape
        x_2 = rearrange(x, 'b c h w -> b ( h w ) c ')
        norm2 = self.norm2(x_2)
        norm2 = rearrange(norm2, 'b ( h w ) c -> b c h w', h=h, w=w)
        ffn = self.IRFFN(norm2)
        x = x + self.drop_path(ffn)
        return x


class CMTBlock(nn.Module):

    def __init__(self, dim, num_heads=8, ffn_ratio=4.0, relative_pos_embeeding=True, no_distance_pos_embeeding=False, features_size=56, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1.0, num_layers=1, drop_path_rate=[0.1]):
        super(CMTBlock, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.ffn_ratio = ffn_ratio
        self.block_list = nn.ModuleList([CMTLayers(dim=self.dim, ffn_ratio=self.ffn_ratio, relative_pos_embeeding=relative_pos_embeeding, no_distance_pos_embeeding=no_distance_pos_embeeding, features_size=features_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio, drop_path_rate=drop_path_rate[i]) for i in range(num_layers)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x


class CMTStem(nn.Module):
    """make the model conv stem module
    """

    def __init__(self, kernel_size, in_channel, out_channel, layers_num):
        super(CMTStem, self).__init__()
        self.layers_num = layers_num
        self.conv3x3_gelu_bn_downsample = ConvGeluBN(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride_size=make_pairs(2))
        self.conv3x3_gelu_bn_list = nn.ModuleList([ConvGeluBN(kernel_size=kernel_size, in_channel=out_channel, out_channel=out_channel, stride_size=1) for _ in range(self.layers_num)])

    def forward(self, x):
        x = self.conv3x3_gelu_bn_downsample(x)
        for i in range(self.layers_num):
            x = self.conv3x3_gelu_bn_list[i](x)
        return x


class PatchAggregation(nn.Module):
    """down sample the feature resolution, build with conv 2x2 stride 2 
    """

    def __init__(self, in_channel, out_channel, kernel_size=2, stride_size=2):
        super(PatchAggregation, self).__init__()
        self.patch_aggregation = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=make_pairs(kernel_size), stride=make_pairs(stride_size))

    def forward(self, x):
        x = self.patch_aggregation(x)
        return x


class ConvolutionMeetVisionTransformers(nn.Module):

    def __init__(self, input_resolution: 'tuple', ape: 'bool', input_channels: 'int', dims_list: 'list', heads_list: 'list', block_list: 'list', sr_ratio_list: 'list', qkv_bias: 'bool', proj_drop: 'float', attn_drop: 'float', rpe: 'bool', pe_nd: 'bool', ffn_ratio: 'float', num_classes: 'int', drop_path_rate: 'float'=0.1):
        """CMT implementation
        Args:
            input_resolution : (h, w) for image resolution
            ape: absoluate position embeeding (learnable)
            input_channels: images input channel, default 3
            dims_list : a list of each stage dimension
            heads_list : mutil head self-attention heads numbers
            block_list : cmt block numbers for each stage
            sr_ratio_list: k,v reduce ratio for each stage
            qkv_bias    : use bias for qkv embeeding
            proj_drop   : proj layer dropout
            attn_drop   : attention dropout
            rpe         : relative position embeeding (learnable )
            pe_nd       : no distance pos embeeding (learnable)
            ffn_ratio   : ffn up & down dims
            num_classes :  output numclasses
            drop_path_rate: Stochastic depth rate. Default: 0.1
        Return:
            cmt model
        """
        super(ConvolutionMeetVisionTransformers, self).__init__()
        assert input_resolution[0] == input_resolution[1], 'input must be square '
        self.input_resolution = input_resolution
        self.input_channels = input_channels
        self.dims_list = dims_list
        self.heads_list = heads_list
        self.block_list = block_list
        self.sr_ratio_list = sr_ratio_list
        self.ape = ape
        self.rpe = rpe
        self.pe_nd = pe_nd
        self.ffn_ratio = ffn_ratio
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.img_height = self.input_resolution[0]
        self.img_width = self.input_resolution[1]
        self.num_patches = self.img_width // 4 * (self.img_height // 4)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dims_list[1]))
            trunc_normal_(self.absolute_pos_embed, std=0.2)
        features_downsample_raito = [2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5]
        resolution_list = [(self.input_resolution[0] // x) for x in features_downsample_raito]
        None
        self.stem = CMTStem(kernel_size=3, in_channel=self.input_channels, out_channel=dims_list[0], layers_num=2)
        if self.drop_path_rate > 0.0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(block_list))]
        else:
            dpr = [(-1) for _ in sum(block_list)]
        self.pool1 = PatchAggregation(in_channel=dims_list[0], out_channel=dims_list[1])
        self.pool2 = PatchAggregation(in_channel=dims_list[1], out_channel=dims_list[2])
        self.pool3 = PatchAggregation(in_channel=dims_list[2], out_channel=dims_list[3])
        self.pool4 = PatchAggregation(in_channel=dims_list[3], out_channel=dims_list[4])
        self.stage1 = CMTBlock(dim=dims_list[1], num_heads=heads_list[0], relative_pos_embeeding=self.rpe, no_distance_pos_embeeding=self.pe_nd, features_size=resolution_list[1], qkv_bias=self.qkv_bias, attn_drop=self.attn_drop, ffn_ratio=self.ffn_ratio, proj_drop=self.proj_drop, sr_ratio=sr_ratio_list[0], num_layers=block_list[0], drop_path_rate=dpr[:block_list[0]])
        self.stage2 = CMTBlock(dim=dims_list[2], num_heads=heads_list[1], relative_pos_embeeding=self.rpe, no_distance_pos_embeeding=self.pe_nd, features_size=resolution_list[2], qkv_bias=self.qkv_bias, attn_drop=self.attn_drop, ffn_ratio=self.ffn_ratio, proj_drop=self.proj_drop, sr_ratio=sr_ratio_list[1], num_layers=block_list[1], drop_path_rate=dpr[sum(block_list[:1]):sum(block_list[:2])])
        self.stage3 = CMTBlock(dim=dims_list[3], num_heads=heads_list[2], relative_pos_embeeding=self.rpe, no_distance_pos_embeeding=self.pe_nd, features_size=resolution_list[3], qkv_bias=self.qkv_bias, attn_drop=self.attn_drop, ffn_ratio=self.ffn_ratio, proj_drop=self.proj_drop, sr_ratio=sr_ratio_list[2], num_layers=block_list[2], drop_path_rate=dpr[sum(block_list[:2]):sum(block_list[:3])])
        self.stage4 = CMTBlock(dim=dims_list[4], num_heads=heads_list[3], relative_pos_embeeding=self.rpe, no_distance_pos_embeeding=self.pe_nd, features_size=resolution_list[4], qkv_bias=self.qkv_bias, attn_drop=self.attn_drop, ffn_ratio=self.ffn_ratio, proj_drop=self.proj_drop, sr_ratio=sr_ratio_list[3], num_layers=block_list[3], drop_path_rate=dpr[sum(block_list[:3]):sum(block_list[:4])])
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(dims_list[4], 1280)
        self.classifier = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(p=0.1)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        x = self.stem(x)
        x = self.pool1(x)
        if self.ape:
            B, C, H, W = x.shape
            x = rearrange(x, ' b c h w -> b (h w) c ')
            x = x + self.absolute_pos_embed
            x = rearrange(x, ' b (h w) c -> b c h w ', h=H)
        x = self.stage1(x)
        x = self.pool2(x)
        x = self.stage2(x)
        x = self.pool3(x)
        x = self.stage3(x)
        x = self.pool4(x)
        x = self.stage4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.gap(x)
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.fc(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CMTStem,
     lambda: ([], {'kernel_size': 4, 'in_channel': 4, 'out_channel': 4, 'layers_num': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvDW3x3,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvGeluBN,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'stride_size': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidualFeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocalPerceptionUint,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PatchAggregation,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_FlyEgle_CMT_pytorch(_paritybench_base):
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

