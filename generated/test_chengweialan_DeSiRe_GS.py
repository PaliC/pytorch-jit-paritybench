import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
gaussian_renderer = _module
gs_render = _module
pvg_render = _module
lpipsPyTorch = _module
lpips = _module
networks = _module
utils = _module
scene = _module
cameras = _module
dinov2 = _module
dynamic_model = _module
emer_waymo_loader = _module
emernerf_loader = _module
envlight = _module
fit3d = _module
gaussian_model = _module
kittimot_loader = _module
scene_utils = _module
waymo_loader = _module
extract_mask_kitti = _module
extract_mask_waymo = _module
extract_mono_cues_kitti = _module
extract_mono_cues_notr = _module
extract_mono_cues_waymo = _module
waymo_converter = _module
waymo_download = _module
separate = _module
train = _module
camera_utils = _module
feature_extractor = _module
general_utils = _module
graphics_utils = _module
image_utils = _module
loss_utils = _module
sh_utils = _module
system_utils = _module
visualize_gs = _module

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


from torchvision.utils import make_grid


from torchvision.utils import save_image


import numpy as np


import warnings


import math


import torch.nn as nn


from typing import Sequence


from itertools import chain


from torchvision import models


from collections import OrderedDict


import random


import logging


from torch import nn


from enum import Enum


from functools import partial


from typing import Tuple


from typing import Union


from typing import Callable


from typing import Optional


from typing import List


from typing import Dict


from typing import Any


from typing import cast


from typing import Type


from torch import Tensor


import torch.utils.checkpoint


from torch.nn.init import trunc_normal_


from typing import Iterable


from typing import TypeVar


import itertools


from functools import reduce


from torch.nn import functional as F


import types


from random import randint


import matplotlib.pyplot as plt


from sklearn.decomposition import PCA


from typing import Literal


from torchvision import transforms


import torchvision.transforms.functional as transF


from collections import defaultdict


import torch.nn.modules.utils as nn_utils


from matplotlib import cm


from typing import NamedTuple


from typing import Iterator


from typing import Mapping


from torch.autograd import Variable


from math import exp


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
        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
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
    old_state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location=None if torch.cuda.is_available() else torch.device('cpu'))
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
        return torch.sum(torch.cat(res, 0), 0, True)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrixCenterShift(znear, zfar, cx, cy, fx, fy, w, h):
    top = cy / fy * znear
    bottom = -(h - cy) / fy * znear
    left = -(w - cx) / fx * znear
    right = cx / fx * znear
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class Camera(nn.Module):

    def __init__(self, colmap_id, R, T, FoVx=None, FoVy=None, cx=None, cy=None, fx=None, fy=None, image=None, image_name=None, uid=0, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device='cuda', timestamp=0.0, resolution=None, image_path=None, pts_depth=None, sky_mask=None, dynamic_mask=None, normal_map=None, ncc_scale=1.0):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image = image
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.resolution = resolution
        self.image_path = image_path
        self.ncc_scale = ncc_scale
        self.nearest_id = []
        self.nearest_names = []
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            None
            None
            self.data_device = torch.device('cuda')
        self.original_image = image.clamp(0.0, 1.0)
        self.sky_mask = sky_mask > 0 if sky_mask is not None else sky_mask
        self.pts_depth = pts_depth if pts_depth is not None else pts_depth
        self.dynamic_mask = dynamic_mask > 0 if dynamic_mask is not None else dynamic_mask
        self.normal_map = normal_map if normal_map is not None else normal_map
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.zfar = 1000.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        if cx is not None:
            self.FoVx = 2 * math.atan(0.5 * self.image_width / fx) if FoVx is None else FoVx
            self.FoVy = 2 * math.atan(0.5 * self.image_height / fy) if FoVy is None else FoVy
            self.projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, cx, cy, fx, fy, self.image_width, self.image_height).transpose(0, 1)
        else:
            self.cx = self.image_width / 2
            self.cy = self.image_height / 2
            self.fx = self.image_width / (2 * np.tan(self.FoVx * 0.5))
            self.fy = self.image_height / (2 * np.tan(self.FoVy * 0.5))
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.timestamp = timestamp
        self.grid = kornia.utils.create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device='cuda')[0]

    def get_world_directions(self, train=False):
        u, v = self.grid.unbind(-1)
        if train:
            directions = torch.stack([(u - self.cx + torch.rand_like(u)) / self.fx, (v - self.cy + torch.rand_like(v)) / self.fy, torch.ones_like(u)], dim=0)
        else:
            directions = torch.stack([(u - self.cx + 0.5) / self.fx, (v - self.cy + 0.5) / self.fy, torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions

    def get_image(self):
        original_image = self.original_image
        image_gray = original_image[0:1] * 0.299 + original_image[1:2] * 0.587 + original_image[2:3] * 0.114
        return original_image, image_gray

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.fx / scale, 0, self.cx / scale], [0, self.fy / scale, self.cy / scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0, 1).contiguous()
        return intrinsic_matrix, extrinsic_matrix

    def get_rays(self, scale=1.0):
        W, H = int(self.image_width / scale), int(self.image_height / scale)
        ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack([(ix - self.cx / scale) / self.fx * scale, (iy - self.cy / scale) / self.fy * scale, torch.ones_like(ix)], -1).float()
        return rays_d

    def get_k(self, scale=1.0):
        K = torch.tensor([[self.fx / scale, 0, self.cx / scale], [0, self.fy / scale, self.cy / scale], [0, 0, 1]])
        return K

    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale / self.fx, 0, -self.cx / self.fx], [0, scale / self.fy, -self.cy / self.fy], [0, 0, 1]])
        return K_T


class Mlp(nn.Module):

    def __init__(self, in_features: 'int', hidden_features: 'Optional[int]'=None, out_features: 'Optional[int]'=None, act_layer: 'Callable[..., nn.Module]'=nn.GELU, drop: 'float'=0.0, bias: 'bool'=True) ->None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return x, x


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(self, img_size: 'Union[int, Tuple[int, int]]'=224, patch_size: 'Union[int, Tuple[int, int]]'=16, in_chans: 'int'=3, embed_dim: 'int'=768, norm_layer: 'Optional[Callable]'=None, flatten_embedding: 'bool'=True) ->None:
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1]
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: 'Tensor') ->Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, f'Input image height {H} is not a multiple of patch height {patch_H}'
        assert W % patch_W == 0, f'Input image width {W} is not a multiple of patch width: {patch_W}'
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x

    def flops(self) ->float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwiGLUFFN(nn.Module):

    def __init__(self, in_features: 'int', hidden_features: 'Optional[int]'=None, out_features: 'Optional[int]'=None, act_layer: 'Optional[Callable[..., nn.Module]]'=None, drop: 'float'=0.0, bias: 'bool'=True) ->None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: 'Tensor') ->Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Attention(nn.Module):

    def __init__(self, dim: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=False, proj_bias: 'bool'=True, attn_drop: 'float'=0.0, proj_drop: 'float'=0.0) ->None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: 'Tensor') ->Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):

    def forward(self, x: 'Tensor', attn_bias=None) ->Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError('xFormers is required for using nested tensors')
            return super().forward(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):

    def __init__(self, dim: 'int', init_values: 'Union[float, Tensor]'=1e-05, inplace: 'bool'=False) ->None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: 'Tensor') ->Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_add_residual_stochastic_depth(x: 'Tensor', residual_func: 'Callable[[Tensor], Tensor]', sample_drop_ratio: 'float'=0.0) ->Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual, alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class Block(nn.Module):

    def __init__(self, dim: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, qkv_bias: 'bool'=False, proj_bias: 'bool'=True, ffn_bias: 'bool'=True, drop: 'float'=0.0, attn_drop: 'float'=0.0, init_values=None, drop_path: 'float'=0.0, act_layer: 'Callable[..., nn.Module]'=nn.GELU, norm_layer: 'Callable[..., nn.Module]'=nn.LayerNorm, attn_class: 'Callable[..., nn.Module]'=Attention, ffn_layer: 'Callable[..., nn.Module]'=Mlp) ->None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: 'Tensor') ->Tensor:

        def attn_residual_func(x: 'Tensor') ->Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: 'Tensor') ->Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual, alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(x, brange, residual, scaling=scaling_vector, alpha=residual_scale_factor)
    return x_plus_residual


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias
    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)
    return attn_bias_cache[all_shapes], cat_tensors


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def drop_add_residual_stochastic_depth_list(x_list: 'List[Tensor]', residual_func: 'Callable[[Tensor, Any], Tensor]', sample_drop_ratio: 'float'=0.0, scaling_vector=None) ->List[Tensor]:
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))
    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):

    def forward_nested(self, x_list: 'List[Tensor]') ->List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)
        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: 'Tensor', attn_bias=None) ->Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: 'Tensor', attn_bias=None) ->Tensor:
                return self.mlp(self.norm2(x))
            x_list = drop_add_residual_stochastic_depth_list(x_list, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio, scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None)
            x_list = drop_add_residual_stochastic_depth_list(x_list, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio, scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None)
            return x_list
        else:

            def attn_residual_func(x: 'Tensor', attn_bias=None) ->Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: 'Tensor', attn_bias=None) ->Tensor:
                return self.ls2(self.mlp(self.norm2(x)))
            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError('xFormers is required for using nested tensors')
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


class BlockChunk(nn.ModuleList):

    def forward(self, x):
        for b in self:
            x = b(x)
        return x


def init_weights_vit_timm(module: 'nn.Module', name: 'str'=''):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


logger = logging.getLogger('dinov2')


def named_apply(fn: 'Callable', module: 'nn.Module', name='', depth_first=True, include_root=False) ->nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DinoVisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, ffn_bias=True, proj_bias=True, drop_path_rate=0.0, drop_path_uniform=False, init_values=None, embed_layer=PatchEmbed, act_layer=nn.GELU, block_fn=NestedTensorBlock, ffn_layer='mlp', block_chunks=1, num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if ffn_layer == 'mlp':
            logger.info('using MLP layer as FFN')
            ffn_layer = Mlp
        elif ffn_layer == 'swiglufused' or ffn_layer == 'swiglu':
            logger.info('using SwiGLU layer as FFN')
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == 'identity':
            logger.info('using Identity layer as FFN')

            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError
        blocks_list = [block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer, init_values=init_values) for i in range(depth)]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i:i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-06)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-06)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs['scale_factor'] = sx, sy
        else:
            kwargs['size'] = w0, h0
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2), mode='bicubic', antialias=self.interpolate_antialias, **kwargs)
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)
        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append({'x_norm_clstoken': x_norm[:, 0], 'x_norm_regtokens': x_norm[:, 1:self.num_register_tokens + 1], 'x_norm_patchtokens': x_norm[:, self.num_register_tokens + 1:], 'x_prenorm': x, 'masks': masks})
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {'x_norm_clstoken': x_norm[:, 0], 'x_norm_regtokens': x_norm[:, 1:self.num_register_tokens + 1], 'x_norm_patchtokens': x_norm[:, self.num_register_tokens + 1:], 'x_prenorm': x, 'masks': masks}

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(cast(BlockChunk, self.blocks[-1]))
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def get_intermediate_layers(self, x: 'torch.Tensor', n: 'Union[int, Sequence]'=1, reshape: 'bool'=False, return_class_token: 'bool'=False, norm=True) ->Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous() for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret['x_norm_clstoken'])


def process_image(image, stride, transforms):
    transformed = transforms(image=np.array(image))
    image_tensor = torch.tensor(transformed['image'])
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    h, w = image_tensor.shape[2:]
    height_int = (h + stride - 1) // stride * stride
    width_int = (w + stride - 1) // stride * stride
    image_resized = torch.nn.functional.interpolate(image_tensor, size=(height_int, width_int), mode='bilinear')
    return image_resized


def Dinov2RegExtractor(original_model, fine_model, image, transforms=None, kmeans=20, only_fine_feats: 'bool'=False):
    """
    Run the demo for a given model option and image
    model_option: ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']
    image_path: path to the image
    kmeans: number of clusters for kmeans. Default is 20. -1 means no kmeans.
    """
    p = original_model.patch_embed.patch_size
    stride = p if isinstance(p, int) else p[0]
    image = image.cpu().numpy()
    image_array = (image * 255).astype(np.uint8)
    image_array = image_array.squeeze(0).transpose(1, 2, 0)
    image = Image.fromarray(image_array)
    fine_feats = None
    ori_feats = None
    if transforms is not None:
        image_resized = process_image(image, stride, transforms)
    else:
        image_resized = image
    with torch.no_grad():
        fine_feats = fine_model.get_intermediate_layers(image_resized, n=[8, 9, 10, 11], reshape=True, return_class_token=False, norm=True)
        if not only_fine_feats:
            ori_feats = original_model.get_intermediate_layers(image_resized, n=[8, 9, 10, 11], reshape=True, return_class_token=False, norm=True)
    fine_feats = fine_feats[-1]
    if not only_fine_feats:
        ori_feats = ori_feats[-1]
    return ori_feats, fine_feats


def GetDinov2RegFeats(original_model, fine_model, image, transforms=None, kmeans=20):
    ori_feats, fine_feats = Dinov2RegExtractor(original_model, fine_model, image, transforms)
    return fine_feats


T = TypeVar('T')


def assert_not_none(value: 'Optional[T]') ->T:
    assert value is not None
    return value


def convert_image_dtype(image: 'np.ndarray', dtype) ->np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f'cannot convert image from {image.dtype} to {dtype}')


def dino_downsample(x, max_size=None):
    if max_size is None:
        return x
    h, w = x.shape[2:]
    if max_size < h or max_size < w:
        scale_factor = min(max_size / x.shape[-2], max_size / x.shape[-1])
        nh = int(h * scale_factor)
        nw = int(w * scale_factor)
        nh = (nh + 13) // 14 * 14
        nw = (nw + 13) // 14 * 14
        x = F.interpolate(x, size=(nh, nw), mode='bilinear')
    return x


def get_intermediate_layers(self, x: 'torch.Tensor', n=1, reshape: 'bool'=False, return_prefix_tokens: 'bool'=False, return_class_token: 'bool'=False, norm: 'bool'=True):
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens:] for out in outputs]
    if reshape:
        B, C, H, W = x.shape
        grid_size = (H - self.patch_embed.patch_size[0]) // self.patch_embed.proj.stride[0] + 1, (W - self.patch_embed.patch_size[1]) // self.patch_embed.proj.stride[1] + 1
        outputs = [out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous() for out in outputs]
    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)


def _ssim_parts(img1, img2, window_size=11):
    sigma = 1.5
    channel = img1.size(-3)
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma1 = torch.sqrt(sigma1_sq.clamp_min(0))
    sigma2 = torch.sqrt(sigma2_sq.clamp_min(0))
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return luminance, contrast, structure


def msssim(x, y, max_size=None, min_size=200):
    raw_orig_size = x.shape[-2:]
    if max_size is not None:
        scale_factor = min(1, max(max_size / x.shape[-2], max_size / x.shape[-1]))
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')
    ssim_maps = list(_ssim_parts(x, y))
    orig_size = x.shape[-2:]
    while x.shape[-2] > min_size and x.shape[-1] > min_size:
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        ssim_maps.extend(tuple(F.interpolate(x, size=orig_size, mode='bilinear') for x in _ssim_parts(x, y)[1:]))
    out = torch.stack(ssim_maps, -1).prod(-1)
    if max_size is not None:
        out = F.interpolate(out, size=raw_orig_size, mode='bilinear')
    return out.mean(1)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim_down(x, y, max_size=None):
    osize = x.shape[2:]
    if max_size is not None:
        scale_factor = max(max_size / x.shape[-2], max_size / x.shape[-1])
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')
    out = ssim(x, y, size_average=False).unsqueeze(1)
    if max_size is not None:
        out = F.interpolate(out, size=osize, mode='bilinear', align_corners=False)
    return out.squeeze(1)


class UncertaintyModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = getattr(dinov2, config.uncertainty_backbone)(pretrained=True)
        self.patch_size = self.backbone.patch_size
        in_features = self.backbone.embed_dim
        self.conv_seg = nn.Sequential(nn.Conv2d(in_features, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
        self.bn = nn.SyncBatchNorm(in_features)
        img_norm_mean = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        img_norm_std = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer('img_norm_mean', img_norm_mean)
        self.register_buffer('img_norm_std', img_norm_std)
        self.max_size = 952 if config.uncertainty_dino_max_size is None else config.uncertainty_dino_max_size
        self._images_cache = {}
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._load_model()

    def _load_model(self):
        original_model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0, dynamic_img_size=True, dynamic_img_pad=False)
        original_model.get_intermediate_layers = types.MethodType(get_intermediate_layers, original_model)
        fine_model = torch.hub.load('ywyue/FiT3D', 'dinov2_reg_small_fine')
        fine_model.get_intermediate_layers = types.MethodType(get_intermediate_layers, fine_model)
        self.original_model = original_model
        self.fine_model = fine_model

    def _get_pad(self, size):
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def _initialize_head_from_checkpoint(self):
        cls_to_ignore = [13, 21, 81, 84]
        backbone = self.config.uncertainty_backbone
        url = f'https://dl.fbaipublicfiles.com/dinov2/{backbone}/{backbone}_ade20k_linear_head.pth'
        with urllib.request.urlopen(url) as f:
            checkpoint_data = f.read()
        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location='cpu')
        old_weight = checkpoint['state_dict']['decode_head.conv_seg.weight']
        new_weight = torch.empty(1, old_weight.shape[1], 1, 1)
        nn.init.normal_(new_weight, 0, 0.0001)
        new_weight[:, cls_to_ignore] = old_weight[:, cls_to_ignore] * 1000
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)
        self.conv_seg.weight.data.copy_(new_weight)
        self.bn.load_state_dict({k[len('decode_head.bn.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('decode_head.bn.')})

    def _get_dino_cached(self, x, cache_entry=None):
        if cache_entry is None or (cache_entry, x.shape) not in self._images_cache:
            with torch.no_grad():
                x = self.backbone.get_intermediate_layers(x, n=[self.backbone.num_heads - 1], reshape=True)[-1]
            if cache_entry is not None:
                self._images_cache[cache_entry, x.shape] = x.detach().cpu()
        else:
            x = self._images_cache[cache_entry, x.shape]
        return x

    def _compute_cosine_similarity(self, x, y, _x_cache=None, _y_cache=None, max_size=None):
        h, w = x.shape[2:]
        if max_size is not None and (max_size < h or max_size < w):
            assert max_size % 14 == 0, 'max_size must be divisible by 14'
            scale_factor = min(max_size / x.shape[-2], max_size / x.shape[-1])
            nh = int(h * scale_factor)
            nw = int(w * scale_factor)
            nh = (nh + 13) // 14 * 14
            nw = (nw + 13) // 14 * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
            y = F.interpolate(y, size=(nh, nw), mode='bilinear')
        x = (x - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        y = (y - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        x = F.pad(x, pads)
        padded_shape = x.shape
        y = F.pad(y, pads)
        with torch.no_grad():
            x = self._get_dino_cached(x, _x_cache)
            y = self._get_dino_cached(y, _y_cache)
        cosine = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        cosine: 'Tensor' = F.interpolate(cosine, size=padded_shape[2:], mode='bilinear', align_corners=False)
        cosine = cosine[:, :, pads[2]:h + pads[2], pads[0]:w + pads[0]]
        if max_size is not None and (max_size < h or max_size < w):
            cosine = F.interpolate(cosine, size=(h, w), mode='bilinear', align_corners=False)
        return cosine.squeeze(1)

    def _forward_uncertainty_features(self, inputs: 'Tensor', _cache_entry=None) ->Tensor:
        transforms = A.Compose([A.Normalize(mean=list(self.img_norm_mean), std=list(self.img_norm_std))])
        x = GetDinov2RegFeats(self.original_model, self.fine_model, inputs, transforms)
        x = F.dropout2d(x, p=self.config.uncertainty_dropout, training=self.training)
        x = self.bn(x)
        logits = self.conv_seg(x)
        logits = logits + math.log(math.exp(1) - 1)
        logits = F.softplus(logits)
        logits: 'Tensor' = F.interpolate(logits, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        logits = logits.clamp(min=self.config.uncertainty_clip_min)
        return logits

    @property
    def device(self):
        return self.img_norm_mean.device

    def forward(self, image: 'Tensor', _cache_entry=None):
        return self._forward_uncertainty_features(image, _cache_entry=_cache_entry)

    def setup_data(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def _load_image(self, img):
        return torch.from_numpy(np.moveaxis(convert_image_dtype(img, np.float32), -1, 0)[None])

    def _scale_input(self, x, max_size: 'Optional[int]'=504):
        h, w = nh, nw = x.shape[2:]
        if max_size is not None:
            scale_factor = min(max_size / x.shape[-2], max_size / x.shape[-1])
            if scale_factor >= 1:
                return x
            nw = int(w * scale_factor)
            nh = int(h * scale_factor)
            nh = (nh + 13) // 14 * 14
            nw = (nw + 13) // 14 * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
        return x

    def _dino_plus_ssim(self, gt, prediction, _cache_entry=None, max_size=None):
        gt_down = dino_downsample(gt, max_size=max_size)
        prediction_down = dino_downsample(prediction, max_size=max_size)
        dino_cosine = self._compute_cosine_similarity(gt_down, prediction_down, _x_cache=_cache_entry).detach()
        dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
        msssim_part = 1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
        return torch.min(dino_part, msssim_part)

    def _compute_losses(self, gt, prediction, sky_mask, prefix='', _cache_entry=None):
        uncertainty = self(self._scale_input(gt, None), _cache_entry=_cache_entry)
        log_uncertainty = torch.log(uncertainty)
        _ssim = ssim_down(gt, prediction, max_size=400).unsqueeze(1)
        _msssim = msssim(gt, prediction, max_size=400, min_size=80).unsqueeze(1)
        if self.config.uncertainty_mode == 'l2reg':
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = 1 / (2 * uncertainty.pow(2))
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == 'l1reg':
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = 1 / uncertainty
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == 'dino':
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=self.max_size)
            prediction_down = dino_downsample(prediction, max_size=self.max_size)
            transforms = A.Compose([A.Normalize(mean=list(self.img_norm_mean), std=list(self.img_norm_std))])
            gt_feats = GetDinov2RegFeats(self.original_model, self.fine_model, gt_down, transforms)
            rendered_feats = GetDinov2RegFeats(self.original_model, self.fine_model, prediction_down, transforms)
            dino_cosine = F.cosine_similarity(gt_feats, rendered_feats, dim=1).unsqueeze(1)
            dino_cosine: 'Tensor' = F.interpolate(dino_cosine, size=gt_down.shape[2:], mode='bilinear', align_corners=False)
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            sky_mask = F.interpolate(sky_mask.float(), size=gt_down.shape[2:], mode='bilinear', align_corners=False).bool()
            dino_part = dino_part.masked_fill(sky_mask, 0.0)
            uncertainty_loss = dino_part * dino_downsample(loss_mult, max_size=self.max_size)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(loss_mult, size=gt.shape[2:], mode='bilinear', align_corners=False)
            if dino_part.shape[2:] != gt.shape[2:]:
                dino_part = F.interpolate(dino_part, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = loss_mult.clamp_max(3)
        elif self.config.uncertainty_mode == 'dino+mssim':
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=self.max_size)
            prediction_down = dino_downsample(prediction, max_size=self.max_size)
            dino_cosine = self._compute_cosine_similarity(gt_down, prediction_down, _x_cache=_cache_entry).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            msssim_part = 1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
            uncertainty_loss = torch.min(dino_part, msssim_part) * dino_downsample(loss_mult, max_size=self.max_size)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(loss_mult, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = loss_mult.clamp_max(3)
        else:
            raise ValueError(f'Invalid uncertainty_mode: {self.config.uncertainty_mode}')
        beta = log_uncertainty.mean()
        loss = uncertainty_loss.mean() + self.config.uncertainty_regularizer_weight * beta
        ssim_discounted = (_ssim * loss_mult).sum() / loss_mult.sum()
        mse = torch.pow(gt - prediction, 2)
        mse_discounted = (mse * loss_mult).sum() / loss_mult.sum()
        psnr_discounted = 10 * torch.log10(1 / mse_discounted)
        metrics = {f'{prefix}loss': loss.item(), f'{prefix}ssim': _ssim.mean().item(), f'{prefix}msssim': _msssim.mean().item(), f'{prefix}ssim_discounted': ssim_discounted.item(), f'{prefix}mse_discounted': mse_discounted.item(), f'{prefix}psnr_discounted': psnr_discounted.item(), f'{prefix}beta': beta.item()}
        return loss, metrics, loss_mult.detach(), dino_part.detach()

    def get_loss(self, gt_image, image, sky_mask, prefix='', _cache_entry=None):
        gt_torch = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)
        sky_mask = sky_mask.unsqueeze(0)
        loss, metrics, loss_mult, dino_part = self._compute_losses(gt_torch, image, sky_mask, prefix, _cache_entry=_cache_entry)
        loss_mult = loss_mult.squeeze(0)
        dino_part = dino_part.squeeze(0)
        metrics[f'{prefix}uncertainty_loss'] = metrics.pop(f'{prefix}loss')
        metrics.pop(f'{prefix}ssim')
        return loss, metrics, loss_mult, dino_part

    @staticmethod
    def load(path: 'str', config: 'OmegaConf') ->'UncertaintyModel':
        ckpt = torch.load(os.path.join(path), map_location='cpu')
        model = UncertaintyModel(config)
        model.load_state_dict(ckpt, strict=False)
        return model

    def save(self, path: 'str'):
        state = self.state_dict()
        state['config'] = OmegaConf.to_yaml(self.config, resolve=True)
        torch.save(state, os.path.join(path))


class Sin(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class ResidualBlock(nn.Sequential):

    def forward(self, input):
        x = super().forward(input)
        minch = min(x.size(1), input.size(1))
        return input[:, :minch] + x[:, :minch]


class EnvLight(torch.nn.Module):

    def __init__(self, resolution=1024):
        super().__init__()
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device='cuda')
        self.base = torch.nn.Parameter(0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True))

    def capture(self):
        return self.base, self.optimizer.state_dict()

    def restore(self, model_args, training_args=None):
        self.base, opt_dict = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)

    def training_setup(self, training_args):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=training_args.envmap_lr, eps=1e-15)

    def forward(self, l):
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:
            l = l.reshape(1, 1, -1, l.shape[-1])
        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)
        return light


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BlockChunk,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {'drop_prob': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerScale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NestedTensorBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SwiGLUFFN,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_chengweialan_DeSiRe_GS(_paritybench_base):
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

