import sys
_module = sys.modules[__name__]
del sys
custom_directives = _module
deploy = _module
server = _module
conf = _module
discrete_unet = _module
ema = _module
model_configs = _module
nn = _module
unet = _module
submitit_train = _module
train = _module
train_arg_parser = _module
data_transform = _module
distributed_mode = _module
edm_time_discretization = _module
eval_loop = _module
grad_scaler = _module
load_and_save = _module
train_loop = _module
data = _module
data = _module
tokenizer = _module
utils = _module
logic = _module
evaluate = _module
flow = _module
generate = _module
state = _module
training = _module
model = _module
rotary = _module
transformer = _module
run_train = _module
eval = _module
run_eval = _module
train = _module
checkpointing = _module
logging = _module
flow_matching = _module
loss = _module
generalized_loss = _module
path = _module
affine = _module
geodesic = _module
mixture = _module
path = _module
path_sample = _module
scheduler = _module
schedule_transform = _module
scheduler = _module
solver = _module
discrete_solver = _module
ode_solver = _module
riemannian_ode_solver = _module
solver = _module
utils = _module
categorical_sampler = _module
manifolds = _module
manifold = _module
sphere = _module
torus = _module
utils = _module
model_wrapper = _module
utils = _module
setup = _module
tests = _module
test_path = _module
test_schedule_transform = _module
test_scheduler = _module
test_discrete_solver = _module
test_ode_solver = _module
test_riemannian_ode_solver = _module
test_utils = _module

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


from typing import Mapping


from typing import Optional


from typing import Tuple


import torch


import torch.nn as nn


import logging


from typing import List


from torch.nn import Module


from torch.nn import Parameter


from torch.nn import ParameterList


import math


import torch as th


from abc import abstractmethod


import numpy as np


import torch.nn.functional as F


import time


import torch.backends.cudnn as cudnn


import torchvision.datasets as datasets


from torchvision.transforms.v2 import Compose


from torchvision.transforms.v2 import RandomHorizontalFlip


from torchvision.transforms.v2 import ToDtype


from torchvision.transforms.v2 import ToImage


import torch.distributed as dist


from typing import Iterable


from torch.nn.modules import Module


from torch.nn.parallel import DistributedDataParallel


from torchvision.utils import save_image


from torch import Tensor


from itertools import chain


from typing import Dict


from torch.utils.data import DataLoader


import itertools


from typing import Any


from torch.utils.data import Dataset


from torch.utils.data import Sampler


from collections import Counter


from torch import nn


from abc import ABC


from torch.nn.modules.loss import _Loss


from torch.optim import Optimizer


from torch.cuda.amp import GradScaler


import torch.multiprocessing as mp


from torch import optim


from torch.nn.parallel import DistributedDataParallel as DDP


from logging import Logger


from torch.func import jvp


from torch.func import vmap


from typing import Union


from math import ceil


from typing import Callable


from torch.nn import functional as F


from typing import Sequence


import abc


class PixelEmbedding(nn.Module):

    def __init__(self, n_tokens: 'int', hidden_size: 'int'):
        super().__init__()
        self.embedding_table = nn.Embedding(n_tokens, hidden_size)

    def forward(self, x: 'torch.Tensor'):
        B, _, H, W = x.shape
        emb = self.embedding_table(x)
        result = emb.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        return result


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * num_spatial ** 2 * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum('bct,bcs->bts', (q * scale).view(bs * self.n_heads, ch, length), (k * scale).view(bs * self.n_heads, ch, length))
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum('bts,bcs->bct', weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        return th.utils.checkpoint.checkpoint(func, *inputs)
    else:
        return func(*inputs)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f'q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}'
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint and self.training)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ConstantEmbedding(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embedding_table = nn.Parameter(torch.empty((1, out_channels)))
        nn.init.uniform_(self.embedding_table, -in_channels ** 0.5, in_channels ** 0.5)

    def forward(self, emb):
        return self.embedding_table.repeat(emb.shape[0], 1)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False, emb_off=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        if emb_off:
            self.emb_layers = ConstantEmbedding(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        else:
            self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint and self.training)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


def base2_fourier_features(inputs: 'torch.Tensor', start: 'int'=0, stop: 'int'=8, step: 'int'=1) ->torch.Tensor:
    freqs = torch.arange(start, stop, step, device=inputs.device, dtype=inputs.dtype)
    w = 2.0 ** freqs * 2 * np.pi
    w = torch.tile(w[None, :], (1, inputs.size(1)))
    h = torch.repeat_interleave(inputs, len(freqs), dim=1)
    h = w[:, :, None, None] * h
    h = torch.cat([torch.sin(h), torch.cos(h)], dim=1)
    return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


logger = logging.getLogger(__name__)


class EMA(Module):

    def __init__(self, model: 'Module', decay: 'float'=0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.register_buffer('num_updates', torch.tensor(0))
        self.shadow_params: 'ParameterList' = ParameterList([Parameter(p.clone().detach(), requires_grad=False) for p in model.parameters() if p.requires_grad])
        self.backup_params: 'List[torch.Tensor]' = []

    def train(self, mode: 'bool') ->None:
        if self.training == mode:
            super().train(mode)
            return
        if not mode:
            logger.info('EMA: Switching from train to eval, backing up parameters and copying EMA params')
            self.backup()
            self.copy_to_model()
        else:
            logger.info('EMA: Switching from eval to train, restoring saved parameters')
            self.restore_to_model()
        super().train(mode)

    def update_ema(self) ->None:
        self.num_updates += 1
        num_updates = self.num_updates.item()
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs) ->torch.Tensor:
        return self.model(*args, **kwargs)

    def copy_to_model(self) ->None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def backup(self) ->None:
        assert self.training, 'Backup can only be created in train mode to avoid backing-up ema weights.'
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)
        else:
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self) ->None:
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)


class SiLU(nn.Module):

    def forward(self, x):
        return x * th.sigmoid(x)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads_channels: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :]
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class Rotary(torch.nn.Module):
    """
    From: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
    """

    def __init__(self, dim: 'int', base: 'int'=10000):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: 'Tensor', seq_dim: 'int'=1) ->Tuple[Tensor, Tensor]:
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return self.cos_cached, self.sin_cached


class LayerNorm(nn.Module):

    def __init__(self, dim: 'int'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: 'Tensor') ->Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: 'int', frequency_embedding_size: 'int'=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: 'Tensor', dim: 'int', max_period: 'int'=10000) ->Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, time: 'Tensor') ->Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def bias_dropout_add_scale(x: 'Tensor', scale: 'Tensor', residual: 'Optional[Tensor]', prob: 'float', training: 'bool') ->Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: 'Tensor', shift: 'Tensor', scale: 'Tensor') ->Tensor:
    return x * (1 + scale) + shift


class DDiTBlock(nn.Module):

    def __init__(self, dim: 'int', n_heads: 'int', cond_dim: 'int', mlp_ratio: 'int'=4, dropout: 'float'=0.1):
        super().__init__()
        assert dim % n_heads == 0, 'dim must be devisable by n_heads'
        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout
        self.head_dim = self.dim // self.n_heads
        self.norm1 = LayerNorm(dim=dim)
        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim, bias=True), nn.GELU(approximate='tanh'), nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: 'Tensor', rotary_cos_sin: 'Tensor', c: 'Tensor') ->Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        x_skip = x
        x = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)
        q, k, v = (item.view(batch_size, seq_len, self.n_heads, self.head_dim) for item in (q, k, v))
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            original_dtype = q.dtype
            q = rotary.apply_rotary_emb_torch(x=q.float(), cos=cos.float(), sin=sin.float())
            k = rotary.apply_rotary_emb_torch(x=k.float(), cos=cos.float(), sin=sin.float())
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        x = F.scaled_dot_product_attention(query=q, key=k, value=v)
        x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)
        x = bias_dropout_add_scale(x=self.attn_out(x), scale=gate_msa, residual=x_skip, prob=self.dropout, training=self.training)
        x = bias_dropout_add_scale(x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)), scale=gate_mlp, residual=x, prob=self.dropout, training=self.training)
        return x


class DDitFinalLayer(nn.Module):

    def __init__(self, hidden_size: 'int', out_channels: 'int', cond_dim: 'int'):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: 'Tensor', c: 'Tensor') ->Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


class Transformer(nn.Module):

    def __init__(self, vocab_size: 'int', masked: 'bool', config: 'DictConfig'):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size
        add_token = 1 if masked else 0
        self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)
        self.time_embedding = TimestepEmbedder(hidden_size=config.cond_dim)
        self.rotary_emb = rotary.Rotary(dim=config.hidden_size // config.n_heads)
        self.blocks = nn.ModuleList([DDiTBlock(dim=config.hidden_size, n_heads=config.n_heads, cond_dim=config.cond_dim, dropout=config.dropout) for _ in range(config.n_blocks)])
        self.output_layer = DDitFinalLayer(hidden_size=config.hidden_size, out_channels=vocab_size + add_token, cond_dim=config.cond_dim)

    def forward(self, x_t: 'Tensor', time: 'Tensor') ->Tensor:
        x = self.vocab_embed(x_t)
        c = F.silu(self.time_embedding(time=time))
        rotary_cos_sin = self.rotary_emb(x=x)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x=x, rotary_cos_sin=rotary_cos_sin, c=c)
            x = self.output_layer(x=x, c=c)
        return x


class MixturePathGeneralizedKL(_Loss):
    """A generalized KL loss for discrete flow matching.
    A class that measures the generalized KL of a discrete flow model :math:`p_{1|t}` w.r.t. a probability path given by ``path``. Note: this class is assuming that the model is trained on the same path.

    For a model trained on a space :math:`\\mathcal{S} = \\mathcal{T}^d`, :math:`\\mathcal{T} = [K] = \\set{1,2,\\ldots,K}`, the loss is given by

    .. math::
            \\ell_i(x_1, x_t, t) = -\\frac{\\dot{\\kappa}_t}{1-\\kappa_t} \\biggr[  p_{1|t}(x_t^i|x_t) -\\delta_{x^i_1}(x_t^i) + (1-\\delta_{x^i_1}(x_t^i))\\left(\\log p_{1|t}(x_1^i|x_t)\\right)\\biggr],

    where :math:`\\kappa_t` is the scheduler associated with ``path``.

    Args:
        path (MixtureDiscreteProbPath): Probability path (x-prediction training).
        reduction (str, optional): Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.
    """

    def __init__(self, path: 'MixtureDiscreteProbPath', reduction: 'str'='mean') ->None:
        super().__init__(None, None, reduction)
        self.path = path

    def forward(self, logits: 'Tensor', x_1: 'Tensor', x_t: 'Tensor', t: 'Tensor') ->Tensor:
        """Evaluates the generalized KL loss.

        Args:
            logits (Tensor): posterior model output (i.e., softmax(``logits``) :math:`=p_{1|t}(x|x_t)`), shape (batch, d, K).
            x_1 (Tensor): target data point :math:`x_1 \\sim q`, shape (batch, d).
            x_t (Tensor): conditional sample at :math:`x_t \\sim p_t(\\cdot|x_1)`, shape (batch, d).
            t (Tensor): times in :math:`[0,1]`, shape (batch).

        Raises:
            ValueError: reduction value must be one of ``'none'`` | ``'mean'`` | ``'sum'``.

        Returns:
            Tensor: Generalized KL loss.
        """
        x_1_shape = x_1.shape
        log_p_1t = torch.log_softmax(logits, dim=-1)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=x_1.unsqueeze(-1))
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=-1, index=x_t.unsqueeze(-1))
        p_1t_xt = p_1t_xt.view(*x_1_shape)
        scheduler_output = self.path.scheduler(t)
        jump_coefficient = (scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t))[(...,) + (None,) * (x_1.dim() - 1)]
        jump_coefficient = jump_coefficient.repeat(1, *x_1_shape[1:])
        delta_x1_xt = x_t == x_1
        loss = -jump_coefficient * (p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'{self.reduction} is not a valid value for reduction')


class Solver(ABC, nn.Module):
    """Abstract base class for solvers."""

    @abstractmethod
    def sample(self, x_0: 'Tensor'=None) ->Tensor:
        ...


class Manifold(nn.Module, metaclass=abc.ABCMeta):
    """A manifold class that contains projection operations and logarithm and exponential maps."""

    @abc.abstractmethod
    def expmap(self, x: 'Tensor', u: 'Tensor') ->Tensor:
        """Computes exponential map :math:`\\exp_x(u)`.

        Args:
            x (Tensor): point on the manifold
            u (Tensor): tangent vector at point :math:`x`

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Tensor: transported point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def logmap(self, x: 'Tensor', y: 'Tensor') ->Tensor:
        """Computes logarithmic map :math:`\\log_x(y)`.

        Args:
            x (Tensor): point on the manifold
            y (Tensor): point on the manifold

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Tensor: tangent vector at point :math:`x`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x: 'Tensor') ->Tensor:
        """Project point :math:`x` on the manifold.

        Args:
            x (Tensor): point to be projected

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Tensor: projected point on the manifold
        """
        raise NotImplementedError

    @abc.abstractmethod
    def proju(self, x: 'Tensor', u: 'Tensor') ->Tensor:
        """Project vector :math:`u` on a tangent space for :math:`x`.

        Args:
            x (Tensor): point on the manifold
            u (Tensor): vector to be projected

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Tensor: projected tangent vector
        """
        raise NotImplementedError


class Euclidean(Manifold):
    """The Euclidean manifold."""

    def expmap(self, x: 'Tensor', u: 'Tensor') ->Tensor:
        return x + u

    def logmap(self, x: 'Tensor', y: 'Tensor') ->Tensor:
        return y - x

    def projx(self, x: 'Tensor') ->Tensor:
        return x

    def proju(self, x: 'Tensor', u: 'Tensor') ->Tensor:
        return u


class Sphere(Manifold):
    """Represents a hyperpshere in :math:`R^D`. Isometric to the product of 1-D spheres."""
    EPS = {torch.float32: 0.0001, torch.float64: 1e-07}

    def expmap(self, x: 'Tensor', u: 'Tensor') ->Tensor:
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.projx(x + u)
        cond = norm_u > self.EPS[norm_u.dtype]
        return torch.where(cond, exp, retr)

    def logmap(self, x: 'Tensor', y: 'Tensor') ->Tensor:
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(self.EPS[x.dtype])
        result = torch.where(cond, u * dist / u.norm(dim=-1, keepdim=True).clamp_min(self.EPS[x.dtype]), u)
        return result

    def projx(self, x: 'Tensor') ->Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    def proju(self, x: 'Tensor', u: 'Tensor') ->Tensor:
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def dist(self, x: 'Tensor', y: 'Tensor', *, keepdim=False) ->Tensor:
        inner = (x * y).sum(-1, keepdim=keepdim)
        return torch.acos(inner)


class FlatTorus(Manifold):
    """Represents a flat torus on the :math:`[0, 2\\pi]^D` subspace. Isometric to the product of 1-D spheres."""

    def expmap(self, x: 'Tensor', u: 'Tensor') ->Tensor:
        return (x + u) % (2 * math.pi)

    def logmap(self, x: 'Tensor', y: 'Tensor') ->Tensor:
        return torch.atan2(torch.sin(y - x), torch.cos(y - x))

    def projx(self, x: 'Tensor') ->Tensor:
        return x % (2 * math.pi)

    def proju(self, x: 'Tensor', u: 'Tensor') ->Tensor:
        return u


class ModelWrapper(ABC, nn.Module):
    """
    This class is used to wrap around another model, adding custom forward pass logic.
    """

    def __init__(self, model: 'nn.Module'):
        super().__init__()
        self.model = model

    def forward(self, x: 'Tensor', t: 'Tensor', **extras) ->Tensor:
        """
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`x` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given x, t
        | returns the model output for input x at time t, with extra information `extra`.

        Args:
            x (Tensor): input data to the model (batch_size, ...).
            t (Tensor): time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns:
            Tensor: model output.
        """
        return self.model(x=x, t=t, **extras)


class DummyModel(ModelWrapper):

    def __init__(self):
        super().__init__(None)

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor', **extras) ->torch.Tensor:
        return (x * 0.0 + 1.0) * 3.0 * t ** 2


class ConstantVelocityModel(ModelWrapper):

    def __init__(self):
        super().__init__(None)
        self.a = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor', **extras) ->torch.Tensor:
        return x * 0.0 + self.a


class HundredVelocityModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return torch.ones_like(x) * 100.0


class ZeroVelocityModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return torch.zeros_like(x)


class ExtraModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, t, must_be_true=False):
        assert must_be_true
        return torch.zeros_like(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConstantEmbedding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConstantVelocityModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DDitFinalLayer,
     lambda: ([], {'hidden_size': 4, 'out_channels': 4, 'cond_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (Downsample,
     lambda: ([], {'channels': 4, 'use_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GroupNorm32,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HundredVelocityModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Rotary,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TimestepBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TimestepEmbedSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {'channels': 4, 'use_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ZeroVelocityModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_flow_matching(_paritybench_base):
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

