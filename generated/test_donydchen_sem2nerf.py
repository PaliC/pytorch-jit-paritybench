import sys
_module = sys.modules[__name__]
del sys
configs = _module
data_configs = _module
paths_config = _module
transforms_config = _module
criteria = _module
lpips = _module
lpips = _module
networks = _module
utils = _module
w_norm = _module
datasets = _module
augmentations = _module
images_dataset = _module
inference_dataset = _module
models = _module
discriminators = _module
discriminator = _module
patch_dis = _module
encoders = _module
helpers = _module
swin_encoder = _module
pigan = _module
model = _module
op = _module
curriculums = _module
math_utils_torch = _module
siren = _module
volumetric_rendering = _module
sem2nerf = _module
options = _module
test_options = _module
train_options = _module
build_celeba_mask = _module
inference3d = _module
train3d = _module
training = _module
coach3d = _module
ranger = _module
common = _module
data_utils = _module
distri_utils = _module
train_utils = _module

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


from typing import Sequence


from itertools import chain


from torchvision import models


from collections import OrderedDict


from torch import nn


import numpy as np


from matplotlib.pyplot import contour


from torch.utils.data import Dataset


import torchvision.transforms as transforms


from torchvision.utils import save_image


import torch.utils.checkpoint as checkpoint


import math


import torch.nn.functional as F


import time


from functools import partial


import matplotlib.pyplot as plt


import random


import matplotlib


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


from time import time


from torch.optim.optimizer import Optimizer


import torch.distributed as dist


from torch.nn import functional as F


from torch import autograd


class LinLayers(nn.ModuleList):

    def __init__(self, n_channels_list: 'Sequence[int]'):
        super(LinLayers, self).__init__([nn.Sequential(nn.Identity(), nn.Conv2d(nc, 1, 1, 1, 0, bias=False)) for nc in n_channels_list])
        for param in self.parameters():
            param.requires_grad = False


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.register_buffer('mean', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('std', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def set_requires_grad(self, state: 'bool'):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: 'torch.Tensor'):
        return (x - self.mean) / self.std

    def forward(self, x: 'torch.Tensor'):
        x = self.z_score(x)
        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class AlexNet(BaseNet):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]
        self.set_requires_grad(False)


class SqueezeNet(BaseNet):

    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]
        self.set_requires_grad(False)


class VGG16(BaseNet):

    def __init__(self):
        super(VGG16, self).__init__()
        self.layers = models.vgg16(True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]
        self.set_requires_grad(False)


def get_network(net_type: 'str'):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


def get_state_dict(net_type: 'str'='alex', version: 'str'='0.1'):
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' + f'master/lpips/weights/v{version}/{net_type}.pth'
    old_state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location='cpu')
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val
    return new_state_dict


class LPIPS(nn.Module):
    """Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, net_type: 'str'='alex', version: 'str'='0.1'):
        assert version in ['0.1'], 'v0.1 is only supported now'
        super(LPIPS, self).__init__()
        self.net = get_network(net_type)
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor'):
        feat_x, feat_y = self.net(x), self.net(y)
        diff = [((fx - fy) ** 2) for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
        return torch.sum(torch.cat(res, 0)) / x.shape[0]


class WNormLoss(nn.Module):

    def __init__(self, start_from_latent_avg=True):
        super(WNormLoss, self).__init__()
        self.start_from_latent_avg = start_from_latent_avg

    def forward(self, latent, latent_avg=None):
        if self.start_from_latent_avg:
            latent = latent - latent_avg
        return torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]


class PatchDiscriminator(nn.Module):

    def __init__(self, size, input_nc=3, ndf=64, hflip=False):
        super(PatchDiscriminator, self).__init__()
        assert size == 32 or size == 64 or size == 128
        self.imsize = size
        self.hflip = hflip
        SN = torch.nn.utils.spectral_norm

        def IN(x):
            return nn.InstanceNorm2d(x)
        blocks = []
        if self.imsize == 128:
            blocks += [SN(nn.Conv2d(input_nc, ndf // 2, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, inplace=True), SN(nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False)), IN(ndf), nn.LeakyReLU(0.2, inplace=True), SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)), IN(ndf * 2), nn.LeakyReLU(0.2, inplace=True)]
        elif self.imsize == 64:
            blocks += [SN(nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, inplace=True), SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)), IN(ndf * 2), nn.LeakyReLU(0.2, inplace=True)]
        else:
            blocks += [SN(nn.Conv2d(input_nc, ndf * 2, 4, 2, 1, bias=False)), IN(ndf * 2), nn.LeakyReLU(0.2, inplace=True)]
        blocks += [SN(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)), IN(ndf * 4), nn.LeakyReLU(0.2, inplace=True), SN(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)), IN(ndf * 8), nn.LeakyReLU(0.2, inplace=True), SN(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))]
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)

    def forward(self, input):
        if self.hflip:
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input), 1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)
        return self.main(input)


def filt_ckpt_keys(ckpt, item_name, model_name):
    assert item_name in ckpt, 'Cannot find [%s] in the checkpoints.' % item_name
    d = ckpt[item_name]
    d_filt = OrderedDict()
    for k, v in d.items():
        k_list = k.split('.')
        if k_list[0] == model_name:
            if k_list[1] == 'module':
                d_filt['.'.join(k_list[2:])] = v
            else:
                d_filt['.'.join(k_list[1:])] = v
    return d_filt


class Discriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, img_size, opts):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.opts = opts
        input_nc = 3
        if opts.use_cond_dis:
            input_nc = input_nc + self.opts.input_nc
        self.input_nc = input_nc
        self.model = self.set_model()
        self.load_weights()

    def set_model(self):
        if self.opts.discriminator_type == 'patch':
            model = PatchDiscriminator(self.img_size, input_nc=self.input_nc)
        else:
            raise Exception('Unknow discriminator_type [%s]' % self.opts.discriminator_type)
        return model

    def set_devices(self, gpu_ids):
        self.device = gpu_ids[0]
        self.model
        if self.opts.distributed_train:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            updated_ckpt = filt_ckpt_keys(ckpt, 'dis_state_dict', 'model')
            self.model.load_state_dict(updated_ckpt, strict=True)
            None

    def forward(self, x, label=None):
        if self.opts.use_cond_dis and label is not None:
            self.x_in = torch.cat([x, label], dim=1)
        else:
            self.x_in = x
        return self.model(self.x_in)


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

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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

    def forward(self, x, return_per_layer_feat=False):
        if return_per_layer_feat:
            layers_feat = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if return_per_layer_feat:
                layers_feat.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if return_per_layer_feat:
            return x, layers_feat
        else:
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


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
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
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(0)

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
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = self.feat_to_out_dict(x)
        return x

    def feat_to_out_dict(self, feat):
        out_dict = {}
        frequencies = feat[:, :feat.shape[-1] // 2]
        phase_shifts = feat[:, feat.shape[-1] // 2:]
        latent_code = [frequencies, phase_shifts]
        out_dict['latent_code'] = latent_code
        return out_dict

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


def fancy_integration(rgb_sigma, z_vals, device, noise_std=0.5, last_back=False, white_back=False, clamp_mode=None, fill_mode=None):
    """Performs NeRF volumetric rendering."""
    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 10000000000.0 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)
    noise = torch.randn(sigmas.shape, device=device) * noise_std
    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * F.softplus(sigmas + noise))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * F.relu(sigmas + noise))
    else:
        raise 'Need to choose clamp mode'
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)
    if last_back:
        weights[:, :, -1] += 1 - weights_sum
    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)
    if white_back:
        rgb_final = rgb_final + 1 - weights_sum
    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1.0, 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)
    return rgb_final, depth_final, weights


def get_sampling_pattern(batch_size, device, resolution_vol, ray_scale_anneal, ray_min_scale, ray_max_scale, ray_random_scale, ray_random_shift, iter):
    w, h = torch.meshgrid([torch.linspace(-1, 1, resolution_vol), torch.linspace(-1, 1, resolution_vol)])
    h = h.repeat(batch_size, 1, 1)
    w = w.repeat(batch_size, 1, 1)
    if ray_scale_anneal > 0:
        k_iter = iter // 1000 * 3
        min_scale = max(ray_min_scale, ray_max_scale * np.exp(-k_iter * ray_scale_anneal))
        min_scale = min(0.9, min_scale)
    else:
        min_scale = ray_min_scale
    scale = 1
    if ray_random_scale:
        scale = torch.Tensor((batch_size,)).uniform_(min_scale, ray_max_scale)
        h = h * scale
        w = w * scale
    if ray_random_shift:
        max_offset = 1 - scale.item()
        index_offset = torch.Tensor((batch_size,)).uniform_(0, max_offset) * (torch.randint(2, (batch_size,)).float() - 0.5) * 2
        h = h + index_offset.view(batch_size, 1, 1).repeat(1, resolution_vol, resolution_vol)
        w = w + index_offset.view(batch_size, 1, 1).repeat(1, resolution_vol, resolution_vol)
    selected_index = torch.stack([h, w], dim=-1)
    return selected_index


def normalize_vecs(vectors: 'torch.Tensor') ->torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / torch.norm(vectors, dim=-1, keepdim=True)


def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end, opts=None):
    """Returns sample points, z_vals, and ray directions in camera space."""
    W, H = resolution
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device), torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan(2 * math.pi * fov / 360 / 2)
    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))
    rays_d_cam = torch.stack(n * [rays_d_cam])
    if opts is not None and opts.patch_train:
        n_points = opts.resolution_vol * opts.resolution_vol
        sample_pattern = get_sampling_pattern(n, device, opts.resolution_vol, opts.ray_scale_anneal, opts.ray_min_scale, opts.ray_max_scale, opts.ray_random_scale, opts.ray_random_shift, opts.iter)
        rays_d_cam = rays_d_cam.reshape(n, W, H, -1).permute(0, 3, 1, 2)
        rays_d_cam = F.grid_sample(rays_d_cam, sample_pattern, mode='bilinear', align_corners=True)
        rays_d_cam = rays_d_cam.permute(0, 2, 3, 1).reshape(n, n_points, -1)
        z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(n_points, 1, 1)
        z_vals = torch.stack(n * [z_vals])
        points = rays_d_cam.unsqueeze(2).repeat(1, 1, num_steps, 1) * z_vals
        return points, z_vals, rays_d_cam, sample_pattern
    else:
        z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W * H, 1, 1)
        points = rays_d_cam[0].unsqueeze(1).repeat(1, num_steps, 1) * z_vals
        z_vals = torch.stack(n * [z_vals])
        points = torch.stack(n * [points])
        return points, z_vals, rays_d_cam


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-05):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)
    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1
    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))
    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)
    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = translation_matrix @ rotation_matrix
    return cam2world


def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:, :, 1:2, :] - z_vals[:, :, 0:1, :]
    offset = (torch.rand(z_vals.shape, device=device) - 0.5) * distance_between_points
    z_vals = z_vals + offset
    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi * 0.5, vertical_mean=math.pi * 0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """
    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean
    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean
    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = (torch.rand((n, 1), device=device) - 0.5) * 2 * v_stddev + v_mean
        v = torch.clamp(v, 1e-05, 1 - 1e-05)
        phi = torch.arccos(1 - 2 * v)
    else:
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean
    phi = torch.clamp(phi, 1e-05, math.pi - 1e-05)
    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r * torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r * torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r * torch.cos(phi)
    return output_points, phi, theta


def transform_sampled_points(points, z_vals, ray_directions, device, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5, mode='normal'):
    """Samples a camera position and maps points in camera space to world space."""
    n, num_rays, num_steps, channels = points.shape
    points, z_vals = perturb_points(points, z_vals, ray_directions, device)
    camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)
    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)
    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)
    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, num_rays, 3)
    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]
    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw


class ImplicitGenerator3d(nn.Module):

    def __init__(self, siren, z_dim, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)
        self.epoch = 0
        self.step = 0

    def set_device(self, device):
        self.device = device
        self.siren.device = device

    def forward(self, latent_code, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, opts, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """
        frequencies, phase_shifts = latent_code
        batch_size = frequencies.shape[0]
        with torch.no_grad():
            if opts.patch_train:
                points_cam, z_vals, rays_d_cam, sample_pattern = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end, opts=opts)
                img_size = opts.resolution_vol
            else:
                points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size * img_size * num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, transformed_ray_directions_expanded)
        coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-05
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies, phase_shifts, transformed_ray_directions_expanded)
            fine_output = fine_output.reshape(batch_size, img_size * img_size, -1, 4)
            all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        out_dict = {'pixels': pixels, 'poses': torch.cat([pitch, yaw], -1)}
        if opts.patch_train:
            out_dict['sample_pattern'] = sample_pattern
        return out_dict

    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""
        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts

    def staged_forward(self, latent_code, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """
        frequencies, phase_shifts = latent_code
        batch_size = frequencies.shape[0]
        self.generate_avg_frequencies()
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size * img_size * num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b + 1, head:tail], frequencies[b:b + 1], phase_shifts[b:b + 1], ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                    head += max_batch_size
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-05
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b + 1, head:tail], frequencies[b:b + 1], phase_shifts[b:b + 1], ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                        head += max_batch_size
                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals
            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode=kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        return pixels, depth_map

    def staged_forward_with_frequencies(self, truncated_frequencies, truncated_phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.7, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        batch_size = truncated_frequencies.shape[0]
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size * img_size * num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b + 1, head:tail], truncated_frequencies[b:b + 1], truncated_phase_shifts[b:b + 1], ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                    head += max_batch_size
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-05
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b + 1, head:tail], truncated_frequencies[b:b + 1], truncated_phase_shifts[b:b + 1], ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
                        head += max_batch_size
                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals
            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode=kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1
        return pixels, depth_map

    def forward_with_frequencies(self, frequencies, phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        batch_size = frequencies.shape[0]
        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size * img_size * num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1
        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-05
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
            all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        return pixels, torch.cat([pitch, yaw], -1)


class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(30.0 * x)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class CustomMappingNetwork(nn.Module):

    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(map_hidden_dim, map_hidden_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(map_hidden_dim, map_hidden_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(map_hidden_dim, map_output_dim))
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]
        return frequencies, phase_shifts


class FiLMLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def frequency_init(freq):

    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init


class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([FiLMLayer(input_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim)])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies * 15 + 30
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = self.color_layer_linear(rbg)
        return torch.cat([rbg, sigma], dim=-1)


class UniformBoxWarp(nn.Module):

    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([FiLMLayer(3, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim)])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies * 15 + 30
        input = self.gridwarper(input)
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        return torch.cat([rbg, sigma], dim=-1)


def modified_first_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    coordinates = coordinates.float()
    grid = grid.float()
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1), coordinates.reshape(batch_size, 1, 1, -1, n_dims), mode='bilinear', padding_mode='zeros', align_corners=True)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H * W * D, C)
    return sampled_features


class EmbeddingPiGAN128(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([FiLMLayer(32 + 3, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim), FiLMLayer(hidden_dim, hidden_dim)])
        None
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96) * 0.01)
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies * 15 + 30
        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        x = torch.cat([shared_features, input], -1)
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        return torch.cat([rbg, sigma], dim=-1)


class EmbeddingPiGAN256(EmbeddingPiGAN128):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 64, 64, 64) * 0.1)


def get_keys(d, name):
    return filt_ckpt_keys(d, 'state_dict', name)


model_paths = {'pigan_celeba': 'pretrained_models/pigan-celeba-pretrained.pth', 'pigan_cat': 'pretrained_models/pigan-cats2-pretrained.pth', 'swin_tiny': 'pretrained_models/swin_tiny_patch4_window7_224.pth'}


class Sem2NeRF(nn.Module):

    def __init__(self, opts):
        super(Sem2NeRF, self).__init__()
        self.set_opts(opts)
        self.encoder = self.set_encoder()
        self.pigan_curriculum = getattr(curriculums, opts.pigan_curriculum_type)
        siren_model = getattr(siren, self.pigan_curriculum['model'])
        self.decoder = getattr(pigan_model, self.pigan_curriculum['generator'])(siren_model, opts.pigan_zdim)
        self.load_weights()

    def update_pigan_curriculum(self):
        with open(self.opts.pigan_steps_conf, 'r') as f:
            train_steps_env = yaml.load(f, Loader=yaml.Loader)
        self.pigan_curriculum.update(train_steps_env)

    def set_devices(self, gpu_ids, mode='train'):
        self.enc_device = gpu_ids[0]
        self.dec_device = gpu_ids[-1]
        self.encoder = self.encoder
        self.decoder = self.decoder
        self.decoder.set_device(self.dec_device)
        if hasattr(self, 'latent_avg'):
            self.latent_avg = [x for x in self.latent_avg]
        if self.opts.distributed_train and mode == 'train':
            self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.enc_device], find_unused_parameters=True)
            self.decoder = torch.nn.parallel.DistributedDataParallel(self.decoder, device_ids=[self.dec_device], find_unused_parameters=True)
        self.running_mode = mode

    def set_encoder(self):
        if self.opts.encoder_type == 'SwinEncoder':
            encoder = swin_encoder.SwinTransformer(img_size=224, patch_size=4, in_chans=self.opts.input_nc, num_classes=512 * 9, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, drop_path_rate=0.2, ape=False, patch_norm=True, use_checkpoint=False)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            None
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            if self.opts.encoder_type in ['SwinEncoder']:
                None
                encoder_ckpt = torch.load(model_paths['swin_tiny'], map_location='cpu')['model']
                if self.opts.label_nc != 0:
                    encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if not ('patch_embed' in k or 'head' in k)}
                else:
                    encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if not 'head' in k}
                self.encoder.load_state_dict(encoder_ckpt, strict=False)
            else:
                raise Exception('Unknown encoder type [%s]' % self.opts.encoder_type)
            None
            if self.opts.pigan_curriculum_type == 'CelebAMask_HQ':
                pigan_model_paths = model_paths['pigan_celeba']
            elif self.opts.pigan_curriculum_type == 'CatMask':
                pigan_model_paths = model_paths['pigan_cat']
            else:
                raise Exception('Cannot find environment %s' % self.opts.pigan_curriculum_type)
            ckpt = torch.load(pigan_model_paths, map_location='cpu')
            self.decoder.load_state_dict(ckpt, strict=False)
            if self.opts.start_from_latent_avg:
                self.__load_latent_avg(ckpt, repeat=1)

    def forward(self, x, pose, return_latents=False, iter_num=0):
        cur_pigan_env = curriculums.extract_metadata(self.pigan_curriculum, iter_num)
        batch_size = cur_pigan_env['batch_size']
        x = x[:batch_size]
        yaw, pitch = pose
        cur_pigan_env['h_mean'] = yaw[:batch_size].unsqueeze(-1)
        cur_pigan_env['v_mean'] = pitch[:batch_size].unsqueeze(-1)
        cur_pigan_env['lock_view_dependence'] = self.opts.pigan_train_lvd
        cur_pigan_env['hierarchical_sample'] = self.opts.pigan_train_hs
        enc_out_dict = self.encoder(x)
        codes = enc_out_dict['latent_code']
        if self.opts.start_from_latent_avg:
            freq, shift = codes
            codes = freq + self.latent_avg[0].repeat(freq.shape[0], 1), shift + self.latent_avg[1].repeat(shift.shape[0], 1)
        if self.opts.patch_train:
            cur_pigan_env['opts'] = {'patch_train': True, 'resolution_vol': cur_pigan_env['resolution_vol'], 'ray_scale_anneal': self.opts.ray_scale_anneal, 'ray_min_scale': self.opts.ray_min_scale, 'ray_max_scale': self.opts.ray_max_scale, 'ray_random_scale': True, 'ray_random_shift': not self.opts.no_ray_random_shift, 'iter': iter_num}
        else:
            cur_pigan_env['opts'] = {'patch_train': False}
            cur_pigan_env['img_size'] = cur_pigan_env['resolution_vol']
        cur_pigan_env['opts'] = Namespace(**cur_pigan_env['opts'])
        self.cur_pigan_env = cur_pigan_env
        dec_out_dict = self.decoder(codes, **cur_pigan_env)
        out_dict = {'images': dec_out_dict['pixels']}
        if return_latents:
            out_dict['latents'] = codes, (self.latent_avg[0].repeat(codes[0].shape[0], 1), self.latent_avg[1].repeat(codes[1].shape[0], 1))
        if self.opts.patch_train:
            out_dict['sample_pattern'] = dec_out_dict['sample_pattern']
        return out_dict

    def forward_inference(self, x, pose, latent_mask=None, inject_latent=None, return_latents=False):
        cur_pigan_env = curriculums.extract_metadata(self.pigan_curriculum, 0)
        cur_pigan_env['batch_size'] = self.opts.test_batch_size
        cur_pigan_env['num_steps'] = self.opts.pigan_infer_ray_step
        cur_pigan_env['img_size'] = self.opts.test_output_size
        cur_pigan_env['max_batch_size'] = self.opts.pigan_infer_max_batch
        cur_pigan_env['lock_view_dependence'] = True
        cur_pigan_env['hierarchical_sample'] = True
        cur_pigan_env['last_back'] = True
        x = x
        enc_out_dict = self.encoder(x)
        codes = enc_out_dict['latent_code']
        if self.opts.start_from_latent_avg:
            freq, shift = codes
            codes = freq + self.latent_avg[0].repeat(freq.shape[0], 1), shift + self.latent_avg[1].repeat(shift.shape[0], 1)
        if latent_mask is not None:
            freq, shift = codes
            inject_freq, inject_shift = inject_latent
            for i in latent_mask:
                start_i = i * self.subnet_attri_getter('decoder').siren.hidden_dim
                end_i = (i + 1) * self.subnet_attri_getter('decoder').siren.hidden_dim
                if inject_latent is not None:
                    freq[:, start_i:end_i] = inject_freq[:, start_i:end_i]
                    shift[:, start_i:end_i] = inject_shift[:, start_i:end_i]
                else:
                    freq[:, start_i:end_i] = 0
                    shift[:, start_i:end_i] = 0
            codes = freq, shift
        if pose is not None:
            yaw, pitch = pose
            cur_pigan_env['h_mean'] = yaw.unsqueeze(-1)
            cur_pigan_env['v_mean'] = pitch.unsqueeze(-1)
        images, _ = self.subnet_attri_getter('decoder').staged_forward(codes, **cur_pigan_env)
        out_dict = {'images': images}
        if return_latents:
            out_dict['latents'] = codes, (self.latent_avg[0].repeat(codes[0].shape[0], 1), self.latent_avg[1].repeat(codes[1].shape[0], 1))
        return out_dict

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'][0], ckpt['latent_avg'][1]
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            z = torch.randn((10000, self.decoder.z_dim))
            with torch.no_grad():
                frequencies, phase_shifts = self.decoder.siren.mapping_network(z)
            avg_frequencies = frequencies.mean(0, keepdim=True)
            avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
            self.latent_avg = avg_frequencies, avg_phase_shifts

    def subnet_attri_getter(self, subnet):
        if self.opts.distributed_train and self.running_mode == 'train':
            return getattr(getattr(self, subnet), 'module')
        else:
            return getattr(self, subnet)

    def requires_grad(self, enc_flag, dec_flag=False):
        requires_grad(self.encoder, enc_flag)
        requires_grad(self.decoder, dec_flag)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (CustomMappingNetwork,
     lambda: ([], {'z_dim': 4, 'map_hidden_dim': 4, 'map_output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FiLMLayer,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchDiscriminator,
     lambda: ([], {'size': 32}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Sine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (UniformBoxWarp,
     lambda: ([], {'sidelength': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_donydchen_sem2nerf(_paritybench_base):
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

