import sys
_module = sys.modules[__name__]
del sys
generate = _module
setup = _module
dataset = _module
ema = _module
model = _module
_attention = _module
unet = _module
scheduler = _module
ddim = _module
utils = _module
train = _module

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


from torchvision import utils


import numpy as np


from torch.utils.data import Dataset


import copy


from torch import nn


from torch import einsum


import math


from torch.nn import functional as F


import matplotlib.pyplot as plt


import random


from torch.nn.utils import clip_grad_norm_


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


import torchvision.transforms as transforms


import pandas as pd


class RMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * x.shape[1] ** 0.5


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, use_flash_attn=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.use_flash_attn = use_flash_attn
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        if not self.use_flash_attn:
            q = q * self.scale
            sim = einsum('b h d i, b h d j -> b h i j', q, k)
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h d j -> b h i d', attn, v)
            out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
            out = rearrange(out, 'b h d (x y) -> b (h d) x y', x=h, y=w)
        return self.to_out(out) + x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, temb_channels, kernel_size=3, stride=1, padding=1, groups=8):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_proj = nn.Sequential(nn.SiLU(), torch.nn.Linear(temb_channels, out_channels))
        self.residual_conv = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.nonlinearity = nn.SiLU()

    def forward(self, x, temb):
        residual = self.residual_conv(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        temb = self.time_emb_proj(temb)
        x += temb[:, :, None, None]
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        return x + residual


def get_attn_layer(in_dim, use_full_attn, use_flash_attn):
    if use_full_attn:
        return Attention(in_dim, use_flash_attn=use_flash_attn)
    else:
        return nn.Identity()


def get_downsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2), nn.Conv2d(in_dim * 4, hidden_dim, 1))
    else:
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)


def get_upsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_dim, hidden_dim, 3, padding=1))
    else:
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)


def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - 1.0)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class UNet(nn.Module):

    def __init__(self, in_channels, hidden_dims=[64, 128, 256, 512], image_size=64, use_flash_attn=False):
        super(UNet, self).__init__()
        self.sample_size = image_size
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        timestep_input_dim = hidden_dims[0]
        time_embed_dim = timestep_input_dim * 4
        self.time_embedding = nn.Sequential(nn.Linear(timestep_input_dim, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))
        self.init_conv = nn.Conv2d(in_channels, out_channels=hidden_dims[0], kernel_size=3, stride=1, padding=1)
        down_blocks = []
        in_dim = hidden_dims[0]
        for idx, hidden_dim in enumerate(hidden_dims[1:]):
            is_last = idx >= len(hidden_dims) - 2
            is_first = idx == 0
            use_attn = True if use_flash_attn else not is_first
            down_blocks.append(nn.ModuleList([ResidualBlock(in_dim, in_dim, time_embed_dim), ResidualBlock(in_dim, in_dim, time_embed_dim), get_attn_layer(in_dim, use_attn, use_flash_attn), get_downsample_layer(in_dim, hidden_dim, is_last)]))
            in_dim = hidden_dim
        self.down_blocks = nn.ModuleList(down_blocks)
        mid_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)
        up_blocks = []
        in_dim = mid_dim
        for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
            is_last = idx >= len(hidden_dims) - 2
            use_attn = True if use_flash_attn else not is_last
            up_blocks.append(nn.ModuleList([ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim), ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim), get_attn_layer(in_dim, use_attn, use_flash_attn), get_upsample_layer(in_dim, hidden_dim, is_last)]))
            in_dim = hidden_dim
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_block = ResidualBlock(hidden_dims[0] * 2, hidden_dims[0], time_embed_dim)
        self.conv_out = nn.Conv2d(hidden_dims[0], out_channels=3, kernel_size=1)

    def forward(self, sample, timesteps):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        timesteps = torch.flatten(timesteps)
        timesteps = timesteps.broadcast_to(sample.shape[0])
        t_emb = sinusoidal_embedding(timesteps, self.hidden_dims[0])
        t_emb = self.time_embedding(t_emb)
        x = self.init_conv(sample)
        r = x.clone()
        skips = []
        for block1, block2, attn, downsample in self.down_blocks:
            x = block1(x, t_emb)
            skips.append(x)
            x = block2(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        for block1, block2, attn, upsample in self.up_blocks:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, t_emb)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block2(x, t_emb)
            x = attn(x)
            x = upsample(x)
        x = self.out_block(torch.cat((x, r), dim=1), t_emb)
        out = self.conv_out(x)
        return {'sample': out}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_filipbasara0_simple_diffusion(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

