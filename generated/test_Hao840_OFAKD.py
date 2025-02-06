import sys
_module = sys.modules[__name__]
del sys
custom_forward = _module
beit = _module
convnext = _module
cycle_mlp = _module
efficientnet = _module
hire_mlp = _module
mlp_mixer = _module
mobilenetv1 = _module
registry = _module
resnet = _module
swin_transformer = _module
vip = _module
vision_transformer = _module
custom_model = _module
beitv2 = _module
cycle_mlp = _module
hire_mlp = _module
mobilenetv1 = _module
vip = _module
distillers = _module
_base = _module
correlation = _module
crd = _module
dist = _module
dkd = _module
fitnet = _module
kd = _module
ofa = _module
rkd = _module
utils = _module
train = _module
validate = _module

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


import torch.nn.functional as F


import math


import torch.nn as nn


from torch import Tensor


from torch.nn import init


from torch.nn.modules.utils import _pair


from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv


from torch import nn


import numpy as np


import logging


import time


from collections import OrderedDict


import torchvision.datasets


import torchvision.utils


from torch.nn.parallel import DistributedDataParallel as NativeDDP


import torch.nn.parallel


import torchvision


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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


class CycleFC(nn.Module):
    """
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size, stride: 'int'=1, padding: 'int'=0, dilation: 'int'=1, groups: 'int'=1, bias: 'bool'=True):
        super(CycleFC, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels * 2, 1, 1)
        start_idx = self.kernel_size[0] * self.kernel_size[1] // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - self.kernel_size[1] // 2
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - self.kernel_size[0] // 2
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: 'Tensor') ->Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def extra_repr(self) ->str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class CycleMLP(nn.Module):

    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CycleBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedOverlapping(nn.Module):
    """ 2D Image to Patch Embedding with overlapping
    """

    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, groups=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class WeightedPermuteMLP(nn.Module):

    def __init__(self, dim, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.segment_dim = segment_dim
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)
        c = self.mlp_c(x)
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3.0, qkv_bias=False, qk_scale=None, attn_drop=0, drop_path_rate=0.0, skip_lam=1.0, mlp_fn=WeightedPermuteMLP, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)
    return blocks


class CycleNet(nn.Module):
    """ CycleMLP Network """

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, mlp_fn=CycleMLP, fork_feat=False):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, CycleFC):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        if self.fork_feat:
            return outs
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean(1))
        return cls_out


class HireMLP(nn.Module):

    def __init__(self, dim, attn_drop=0.0, proj_drop=0.0, pixel=2, step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        """
        self.pixel: h and w in inner-region rearrangement
        self.step: s in cross-region rearrangement
        """
        self.pixel = pixel
        self.step = step
        self.step_pad_mode = step_pad_mode
        self.pixel_pad_mode = pixel_pad_mode
        self.mlp_h1 = nn.Conv2d(dim * pixel, dim // 2, 1, bias=False)
        self.mlp_h1_norm = nn.BatchNorm2d(dim // 2)
        self.mlp_h2 = nn.Conv2d(dim // 2, dim * pixel, 1, bias=True)
        self.mlp_w1 = nn.Conv2d(dim * pixel, dim // 2, 1, bias=False)
        self.mlp_w1_norm = nn.BatchNorm2d(dim // 2)
        self.mlp_w2 = nn.Conv2d(dim // 2, dim * pixel, 1, bias=True)
        self.mlp_c = nn.Conv2d(dim, dim, 1, bias=True)
        self.act = nn.ReLU()
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """
        B, C, H, W = x.shape
        pad_h, pad_w = (self.pixel - H % self.pixel) % self.pixel, (self.pixel - W % self.pixel) % self.pixel
        h, w = x.clone(), x.clone()
        if self.step:
            if self.step_pad_mode == '0':
                h = F.pad(h, (0, 0, self.step, 0), 'constant', 0)
                w = F.pad(w, (self.step, 0, 0, 0), 'constant', 0)
                h = torch.narrow(h, 2, 0, H)
                w = torch.narrow(w, 3, 0, W)
            elif self.step_pad_mode == 'c':
                h = torch.roll(h, self.step, -2)
                w = torch.roll(w, self.step, -1)
            else:
                raise NotImplementedError('Invalid pad mode.')
        if self.pixel_pad_mode == '0':
            h = F.pad(h, (0, 0, 0, pad_h), 'constant', 0)
            w = F.pad(w, (0, pad_w, 0, 0), 'constant', 0)
        elif self.pixel_pad_mode == 'c':
            h = F.pad(h, (0, 0, 0, pad_h), mode='circular')
            w = F.pad(w, (0, pad_w, 0, 0), mode='circular')
        elif self.pixel_pad_mode == 'replicate':
            h = F.pad(h, (0, 0, 0, pad_h), mode='replicate')
            w = F.pad(w, (0, pad_w, 0, 0), mode='replicate')
        else:
            raise NotImplementedError('Invalid pad mode.')
        h = h.reshape(B, C, (H + pad_h) // self.pixel, self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C * self.pixel, (H + pad_h) // self.pixel, W)
        w = w.reshape(B, C, H, (W + pad_w) // self.pixel, self.pixel).permute(0, 1, 4, 2, 3).reshape(B, C * self.pixel, H, (W + pad_w) // self.pixel)
        h = self.mlp_h1(h)
        h = self.mlp_h1_norm(h)
        h = self.act(h)
        h = self.mlp_h2(h)
        w = self.mlp_w1(w)
        w = self.mlp_w1_norm(w)
        w = self.act(w)
        w = self.mlp_w2(w)
        h = h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C, H + pad_h, W)
        w = w.reshape(B, C, self.pixel, H, (W + pad_w) // self.pixel).permute(0, 1, 3, 4, 2).reshape(B, C, H, W + pad_w)
        h = torch.narrow(h, 2, 0, H)
        w = torch.narrow(w, 3, 0, W)
        if self.step and self.step_pad_mode == 'c':
            h = torch.roll(h, -self.step, -2)
            w = torch.roll(w, -self.step, -1)
        c = self.mlp_c(x)
        a = (h + w + c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HireBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0, pixel=2, step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = HireMLP(dim, attn_drop=attn_drop, pixel=pixel, step=step, step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HireMLPNet(nn.Module):

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None, mlp_ratios=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, pixel=[2, 2, 2, 2], step_stride=[2, 2, 2, 2], step_dilation=[1, 1, 1, 1], step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, pixel=pixel[i], step_stride=step_stride[i], step_dilation=step_dilation[i], step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))
        self.network = nn.ModuleList(network)
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        cls_out = self.head(x.flatten(2).mean(2))
        return cls_out


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1), nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model[3][:-1](self.model[0:3](x))
        x = self.model[5][:-1](self.model[4:5](F.relu(x)))
        x = self.model[11][:-1](self.model[6:11](F.relu(x)))
        x = self.model[13][:-1](self.model[12:13](F.relu(x)))
        x = self.model[14](F.relu(x))
        x = x.reshape(-1, 1024)
        out = self.fc(x)
        return out


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x


class VisionPermutator(nn.Module):
    """ Vision Permutator
    """

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, mlp_fn=WeightedPermuteMLP):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))
        self.network = nn.ModuleList(network)
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))


class BaseDistiller(nn.Module):

    def __init__(self, student, teacher, criterion, args):
        super(BaseDistiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.args = args

    def forward(self, image, label, *args, **kwargs):
        raise NotImplementedError

    def get_learnable_parameters(self):
        student_params = 0
        extra_params = 0
        for n, p in self.named_parameters():
            if n.startswith('student'):
                student_params += p.numel()
            elif n.startswith('teacher'):
                continue
            elif p.requires_grad:
                extra_params += p.numel()
        return student_params, extra_params


class Vanilla(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(Vanilla, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        logits_student = self.student(image)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {'loss_gt': loss_gt}
        return logits_student, losses_dict


class LinearEmbed(nn.Module):

    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class Correlation(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(Correlation, self).__init__(student, teacher, criterion, args)
        feat_s_channel = student.stage_info(-1)[1]
        feat_t_channel = teacher.stage_info(-1)[1]
        self.embed_s = LinearEmbed(feat_s_channel, self.args.correlation_feat_dim)
        self.embed_t = LinearEmbed(feat_t_channel, self.args.correlation_feat_dim)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)
        logits_student, feat_student = self.student(image, requires_feat=True)
        f_s = self.embed_s(feat_student[-1])
        f_t = self.embed_t(feat_teacher[-1])
        delta = torch.abs(f_s - f_t)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * self.args.correlation_scale * torch.mean((delta[:-1] * delta[1:]).sum(1))
        losses_dict = {'loss_gt': loss_gt, 'loss_kd': loss_kd}
        return logits_student, losses_dict


class ContrastLoss(nn.Module):
    """contrastive loss"""

    def __init__(self, num_data):
        super(ContrastLoss, self).__init__()
        self.num_data = num_data

    def forward(self, x):
        eps = 1e-07
        bsz = x.shape[0]
        m = x.size(1) - 1
        Pn = 1 / float(self.num_data)
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        return loss


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = self.prob[large] - 1.0 + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in (smaller + larger):
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob
        self.alias = self.alias

    def draw(self, N):
        """Draw N samples from multinomial"""
        K = self.alias.size(0)
        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj


class ContrastMemory(nn.Module):
    """memory buffer that supplies large amount of negative samples."""

    def __init__(self, inputSize, output_size, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.n_lem = output_size
        self.unigrams = torch.ones(self.n_lem)
        self.multinomial = AliasMethod(self.unigrams)
        self.K = K
        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(output_size, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(output_size, inputSize).mul_(2 * stdv).add_(-stdv))

    def cuda(self, *args, **kwargs):
        super(ContrastMemory, self)
        self.multinomial

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()
        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)
            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)
        return out_v1, out_v2


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class CRD(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(CRD, self).__init__(student, teacher, criterion, args)
        num_data = kwargs['num_data']
        feat_s_channel = student.stage_info(-1)[1]
        feat_t_channel = teacher.stage_info(-1)[1]
        self.embed_s = Embed(feat_s_channel, self.args.crd_feat_dim)
        self.embed_t = Embed(feat_t_channel, self.args.crd_feat_dim)
        self.contrast = ContrastMemory(self.args.crd_feat_dim, num_data, self.args.crd_k, self.args.crd_temperature, self.args.crd_momentum)
        self.criterion_s = ContrastLoss(num_data)
        self.criterion_t = ContrastLoss(num_data)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)
        logits_student, feat_student = self.student(image, requires_feat=True)
        f_s = self.embed_s(feat_student[-1])
        f_t = self.embed_t(feat_teacher[-1])
        index, contrastive_index = args
        out_s, out_t = self.contrast(f_s, f_t, index, contrastive_index)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * (s_loss + t_loss)
        losses_dict = {'loss_gt': loss_gt, 'loss_kd': loss_kd}
        return logits_student, losses_dict


def cosine_similarity(a, b, eps=1e-08):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-08):
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def dist_loss(logits_student, logits_teacher, beta=1.0, gamma=1.0, temperature=1.0):
    y_s = (logits_student / temperature).softmax(dim=1)
    y_t = (logits_teacher / temperature).softmax(dim=1)
    inter_loss = temperature ** 2 * inter_class_relation(y_s, y_t)
    intra_loss = temperature ** 2 * intra_class_relation(y_s, y_t)
    return beta * inter_loss + gamma * intra_loss


class DIST(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(DIST, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)
        logits_student = self.student(image)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * dist_loss(logits_student, logits_teacher, self.args.dist_beta, self.args.dist_gamma, self.args.dist_tau)
        losses_dict = {'loss_gt': loss_gt, 'loss_kd': loss_kd}
        return logits_student, losses_dict


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * temperature ** 2
    pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)
    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean') * temperature ** 2
    return alpha * tckd_loss + beta * nckd_loss


class DKD(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(DKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)
        logits_student = self.student(image)
        if len(label.shape) == 2:
            target = label.max(1)[1]
        else:
            target = label
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * dkd_loss(logits_student, logits_teacher, target, self.dkd_alpha, self.dkd_beta, self.dkd_temperature)
        losses_dict = {'loss_gt': loss_gt, 'loss_kd': loss_kd}
        return logits_student, losses_dict


def get_module_dict(module_dict, k):
    if not isinstance(k, str):
        k = str(k)
    return module_dict[k]


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def is_cnn_model(distiller):
    if hasattr(distiller, 'module'):
        _, sizes = distiller.module.stage_info(1)
    else:
        _, sizes = distiller.stage_info(1)
    if len(sizes) == 3:
        return True
    elif len(sizes) == 2:
        return False
    else:
        raise RuntimeError('unknown model feature shape')


def set_module_dict(module_dict, k, v):
    if not isinstance(k, str):
        k = str(k)
    module_dict[k] = v


class FitNet(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(FitNet, self).__init__(student, teacher, criterion, args)
        assert is_cnn_model(student) and is_cnn_model(teacher), 'current FitNet implementation only support cnn models!'
        self.projector = nn.ModuleDict()
        for stage in self.args.fitnet_stage:
            _, size_s = self.student.stage_info(stage)
            _, size_t = self.teacher.stage_info(stage)
            in_chans_s, _, _ = size_s
            in_chans_t, _, _ = size_t
            projector = nn.Conv2d(in_chans_s, in_chans_t, 1, 1, 0, bias=False)
            set_module_dict(self.projector, stage, projector)
        self.projector.apply(init_weights)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)
        logits_student, feat_student = self.student(image, requires_feat=True)
        fitnet_losses = []
        for stage in self.args.fitnet_stage:
            idx_s, _ = self.student.stage_info(stage)
            idx_t, _ = self.teacher.stage_info(stage)
            feat_s = get_module_dict(self.projector, stage)(feat_student[idx_s])
            feat_t = feat_teacher[idx_t]
            fitnet_losses.append(F.mse_loss(feat_s, feat_t))
        loss_fitnet = self.args.fitnet_loss_weight * sum(fitnet_losses)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {'loss_gt': loss_gt, 'loss_fitnet': loss_fitnet}
        return logits_student, losses_dict


def kd_loss(logits_student, logits_teacher, temperature=1.0):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
    loss_kd *= temperature ** 2
    return loss_kd


class KD(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(KD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)
        logits_student = self.student(image)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)
        losses_dict = {'loss_gt': loss_gt, 'loss_kd': loss_kd}
        return logits_student, losses_dict


class BKD(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(BKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)
        logits_student = self.student(image)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * F.binary_cross_entropy_with_logits(logits_student, logits_teacher.softmax(1))
        losses_dict = {'loss_gt': loss_gt, 'loss_kd': loss_kd}
        return logits_student, losses_dict


class GAP1d(nn.Module):

    def __init__(self):
        super(GAP1d, self).__init__()

    def forward(self, x):
        return x.mean(1)


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm, act_layer=nn.Identity):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        in_features = 4 * dim
        self.reduction = nn.Linear(in_features, self.out_dim, bias=False)
        self.act = act_layer()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, 'input feature has wrong size')
        _assert(H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.')
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        x = self.act(x)
        return x


class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False), nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(channel_in, affine=affine), nn.ReLU(inplace=False), nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False), nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(channel_out, affine=affine), nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)


class TokenFilter(nn.Module):
    """remove cls tokens in forward"""

    def __init__(self, number=1, inverse=False, remove_mode=True):
        super(TokenFilter, self).__init__()
        self.number = number
        self.inverse = inverse
        self.remove_mode = remove_mode

    def forward(self, x):
        if self.inverse and self.remove_mode:
            x = x[:, :-self.number, :]
        elif self.inverse and not self.remove_mode:
            x = x[:, -self.number:, :]
        elif not self.inverse and self.remove_mode:
            x = x[:, self.number:, :]
        else:
            x = x[:, :self.number, :]
        return x


class TokenFnContext(nn.Module):

    def __init__(self, token_num=0, fn: 'nn.Module'=nn.Identity(), token_fn: 'nn.Module'=nn.Identity(), inverse=False):
        super(TokenFnContext, self).__init__()
        self.token_num = token_num
        self.fn = fn
        self.token_fn = token_fn
        self.inverse = inverse
        self.token_filter = TokenFilter(number=token_num, inverse=inverse, remove_mode=False)
        self.feature_filter = TokenFilter(number=token_num, inverse=inverse)

    def forward(self, x):
        tokens = self.token_filter(x)
        features = self.feature_filter(x)
        features = self.fn(features)
        if self.token_num == 0:
            return features
        tokens = self.token_fn(tokens)
        if self.inverse:
            x = torch.cat([features, tokens], dim=1)
        else:
            x = torch.cat([tokens, features], dim=1)
        return x


def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.0):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(-(prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()


class OFA(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(OFA, self).__init__(student, teacher, criterion, args)
        if len(self.args.ofa_eps) == 1:
            eps = [self.args.ofa_eps[0] for _ in range(len(self.args.ofa_stage) + 1)]
            self.args.ofa_eps = eps
        assert len(self.args.ofa_stage) + 1 == len(self.args.ofa_eps)
        self.projector = nn.ModuleDict()
        is_cnn_student = is_cnn_model(student)
        _, feature_dim_t = self.teacher.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)
        for stage in self.args.ofa_stage:
            _, size_s = self.student.stage_info(stage)
            if is_cnn_student:
                in_chans, _, _ = size_s
                if stage != 4:
                    down_sample_blk_num = 4 - stage
                    down_sample_blks = []
                    for i in range(down_sample_blk_num):
                        if i == down_sample_blk_num - 1:
                            out_chans = max(feature_dim_s, feature_dim_t)
                        else:
                            out_chans = in_chans * 2
                        down_sample_blks.append(SepConv(in_chans, out_chans))
                        in_chans *= 2
                else:
                    down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]
                projector = nn.Sequential(*down_sample_blks, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(max(feature_dim_s, feature_dim_t), args.num_classes))
            else:
                patch_num, embed_dim = size_s
                token_num = getattr(student, 'num_tokens', 0)
                final_patch_grid = 7
                patch_grid = int(patch_num ** 0.5)
                merge_num = max(int(np.log2(patch_grid / final_patch_grid)), 0)
                merger_modules = []
                for i in range(merge_num):
                    if i == 0:
                        merger_modules.append(PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i), dim=embed_dim, out_dim=feature_dim_s, act_layer=nn.GELU))
                    else:
                        merger_modules.append(PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i), dim=feature_dim_s, out_dim=feature_dim_s, act_layer=nn.GELU if i != merge_num - 1 else nn.Identity))
                patch_merger = nn.Sequential(*merger_modules)
                blocks = nn.Sequential(*[Block(dim=feature_dim_s, num_heads=4) for _ in range(max(4 - stage, 1))])
                if token_num != 0:
                    get_feature = nn.Sequential(TokenFilter(token_num, remove_mode=False), nn.Flatten())
                else:
                    get_feature = GAP1d()
                projector = nn.Sequential(TokenFnContext(token_num, patch_merger), blocks, get_feature, nn.Linear(feature_dim_s, args.num_classes))
            set_module_dict(self.projector, stage, projector)
        self.projector.apply(init_weights)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)
        logits_student, feat_student = self.student(image, requires_feat=True)
        num_classes = logits_student.size(-1)
        if len(label.shape) != 1:
            target_mask = F.one_hot(label.argmax(-1), num_classes)
        else:
            target_mask = F.one_hot(label, num_classes)
        ofa_losses = []
        for stage, eps in zip(self.args.ofa_stage, self.args.ofa_eps):
            idx_s, _ = self.student.stage_info(stage)
            feat_s = feat_student[idx_s]
            logits_student_head = get_module_dict(self.projector, stage)(feat_s)
            ofa_losses.append(ofa_loss(logits_student_head, logits_teacher, target_mask, eps, self.args.ofa_temperature))
        loss_ofa = self.args.ofa_loss_weight * sum(ofa_losses)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * ofa_loss(logits_student, logits_teacher, target_mask, self.args.ofa_eps[-1], self.args.ofa_temperature)
        losses_dict = {'loss_gt': loss_gt, 'loss_kd': loss_kd, 'loss_ofa': loss_ofa}
        return logits_student, losses_dict


def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKD(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(RKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)
        logits_student, feat_student = self.student(image, requires_feat=True)
        f_s = feat_student[-1]
        f_t = feat_teacher[-1]
        stu = f_s.view(f_s.shape[0], -1)
        tea = f_t.view(f_t.shape[0], -1)
        with torch.no_grad():
            t_d = _pdist(tea, self.args.rkd_squared, self.args.rkd_eps)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
        d = _pdist(stu, self.args.rkd_squared, self.args.rkd_eps)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss_d = F.smooth_l1_loss(d, t_d)
        with torch.no_grad():
            td = tea.unsqueeze(0) - tea.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        sd = stu.unsqueeze(0) - stu.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss_a = F.smooth_l1_loss(s_angle, t_angle)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * (self.args.rkd_distance_weight * loss_d + self.args.rkd_angle_weight * loss_a)
        losses_dict = {'loss_gt': loss_gt, 'loss_kd': loss_kd}
        return logits_student, losses_dict


class LambdaModule(nn.Module):

    def __init__(self, lambda_fn):
        super(LambdaModule, self).__init__()
        self.fn = lambda_fn

    def forward(self, x):
        return self.fn(x)


class MyPatchMerging(nn.Module):

    def __init__(self, out_patch_num):
        super().__init__()
        self.out_patch_num = out_patch_num

    def forward(self, x):
        B, L, D = x.shape
        patch_size = int(L ** 0.5)
        assert patch_size ** 2 == L
        out_patch_size = int(self.out_patch_num ** 0.5)
        assert out_patch_size ** 2 == self.out_patch_num
        grid_size = patch_size // out_patch_size
        assert grid_size * out_patch_size == patch_size
        x = x.view(B, out_patch_size, grid_size, out_patch_size, grid_size, D)
        x = torch.einsum('bhpwqd->bhwpqd', x)
        x = x.reshape(shape=(B, out_patch_size ** 2, -1))
        return x


def _unpatchify(x, p):
    """
    x: (N, L, patch_size**2 *C)
    imgs: (N, C, H, W)
    """
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
    return imgs


class Unpatchify(nn.Module):

    def __init__(self, p):
        super(Unpatchify, self).__init__()
        self.p = p

    def forward(self, x):
        return _unpatchify(x, self.p)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContrastLoss,
     lambda: ([], {'num_data': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CycleBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CycleMLP,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample,
     lambda: ([], {'in_embed_dim': 4, 'out_embed_dim': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GAP1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaModule,
     lambda: ([], {'lambda_fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (MyPatchMerging,
     lambda: ([], {'out_patch_num': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Normalize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PermutatorBlock,
     lambda: ([], {'dim': 4, 'segment_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SepConv,
     lambda: ([], {'channel_in': 4, 'channel_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TokenFilter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TokenFnContext,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Unpatchify,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Hao840_OFAKD(_paritybench_base):
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

