import sys
_module = sys.modules[__name__]
del sys
app = _module
discord_presence = _module
i18n = _module
scan = _module
installation_checker = _module
Applio = _module
loadThemes = _module
version_checker = _module
core = _module
config = _module
infer = _module
pipeline = _module
algorithm = _module
attentions = _module
commons = _module
discriminators = _module
encoders = _module
hifigan = _module
hifigan_mrf = _module
hifigan_nsf = _module
refinegan = _module
modules = _module
normalization = _module
residuals = _module
synthesizers = _module
F0Extractor = _module
FCPE = _module
RMVPE = _module
analyzer = _module
gdown = _module
launch_tensorboard = _module
model_download = _module
prerequisites_download = _module
pretrained_selector = _module
split_audio = _module
tts = _module
utils = _module
zluda = _module
data_utils = _module
extract = _module
preparing_files = _module
losses = _module
mel_processing = _module
preprocess = _module
slicer = _module
change_info = _module
extract_index = _module
extract_model = _module
model_blender = _module
model_information = _module
train = _module
utils = _module
download = _module
extra = _module
f0_extractor = _module
processing = _module
inference = _module
plugins = _module
plugins_core = _module
report = _module
lang = _module
model_author = _module
presence = _module
restart = _module
themes = _module
version = _module
settings = _module
voice_blender = _module

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


import time


import logging


import numpy as np


import re


import torch.nn.functional as F


from scipy import signal


from torch import Tensor


import math


from typing import Optional


from torch.utils.checkpoint import checkpoint


from torch.nn.utils.parametrizations import spectral_norm


from torch.nn.utils.parametrizations import weight_norm


from torch.nn.utils import remove_weight_norm


from torch import nn


from torch.nn import functional as F


from torch.nn.utils.parametrize import remove_parametrizations


from itertools import chain


from typing import Tuple


from typing import Union


import torch.nn as nn


from torchaudio.transforms import Resample


import torch.utils.data


from functools import partial


from typing import List


import warnings


from collections import OrderedDict


from collections import deque


from random import randint


from random import shuffle


from time import time as ttime


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.utils.data import DataLoader


import torch.distributed as dist


import torch.multiprocessing as mp


import matplotlib.pyplot as plt


def convert_pad_shape(pad_shape):
    """
    Convert the pad shape to a list of integers.

    Args:
        pad_shape: The pad shape..
    """
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention module with optional relative positional encoding and proximal bias.

    Args:
        channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_heads (int): Number of attention heads.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        window_size (int, optional): Window size for relative positional encoding. Defaults to None.
        heads_share (bool, optional): Whether to share relative positional embeddings across heads. Defaults to True.
        block_length (int, optional): Block length for local attention. Defaults to None.
        proximal_bias (bool, optional): Whether to use proximal bias in self-attention. Defaults to False.
        proximal_init (bool, optional): Whether to initialize the key projection weights the same as query projection weights. Defaults to False.
    """

    def __init__(self, channels: 'int', out_channels: 'int', n_heads: 'int', p_dropout: 'float'=0.0, window_size: 'int'=None, heads_share: 'bool'=True, block_length: 'int'=None, proximal_bias: 'bool'=False, proximal_init: 'bool'=False):
        super().__init__()
        assert channels % n_heads == 0, 'Channels must be divisible by the number of heads.'
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)
        if window_size:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = torch.nn.Parameter(torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = torch.nn.Parameter(torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels) * rel_stddev)
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        torch.nn.init.xavier_uniform_(self.conv_o.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = *key.size(), query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size:
            assert t_s == t_t, 'Relative attention only supports self-attention.'
            scores += self._compute_relative_scores(query, t_s)
        if self.proximal_bias:
            assert t_s == t_t, 'Proximal bias only supports self-attention.'
            scores += self._attention_bias_proximal(t_s)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -10000.0)
            if self.block_length:
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -10000.0)
        p_attn = self.drop(torch.nn.functional.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)
        if self.window_size:
            output += self._apply_relative_values(p_attn, t_s)
        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn

    def _compute_relative_scores(self, query, length):
        rel_emb = self._get_relative_embeddings(self.emb_rel_k, length)
        rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), rel_emb)
        return self._relative_position_to_absolute_position(rel_logits)

    def _apply_relative_values(self, p_attn, length):
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        rel_emb = self._get_relative_embeddings(self.emb_rel_v, length)
        return self._matmul_with_relative_values(rel_weights, rel_emb)

    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        start = max(self.window_size + 1 - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            embeddings = torch.nn.functional.pad(embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        return embeddings[:, start:end]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view(batch, heads, length * 2 * length)
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1:]

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view(batch, heads, length ** 2 + length * (length - 1))
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view(batch, heads, length, 2 * length)[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = r.unsqueeze(0) - r.unsqueeze(1)
        return -torch.log1p(torch.abs(diff)).unsqueeze(0).unsqueeze(0)


class FFN(torch.nn.Module):
    """
    Feed-forward network module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        filter_channels (int): Number of filter channels in the convolution layers.
        kernel_size (int): Kernel size of the convolution layers.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        activation (str, optional): Activation function to use. Defaults to None.
        causal (bool, optional): Whether to use causal padding in the convolution layers. Defaults to False.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', filter_channels: 'int', kernel_size: 'int', p_dropout: 'float'=0.0, activation: 'str'=None, causal: 'bool'=False):
        super().__init__()
        self.padding_fn = self._causal_padding if causal else self._same_padding
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)
        self.activation = activation

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding_fn(x * x_mask))
        x = self._apply_activation(x)
        x = self.drop(x)
        x = self.conv_2(self.padding_fn(x * x_mask))
        return x * x_mask

    def _apply_activation(self, x):
        if self.activation == 'gelu':
            return x * torch.sigmoid(1.702 * x)
        return torch.relu(x)

    def _causal_padding(self, x):
        pad_l, pad_r = self.conv_1.kernel_size[0] - 1, 0
        return torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]]))

    def _same_padding(self, x):
        pad = (self.conv_1.kernel_size[0] - 1) // 2
        return torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad, pad]]))


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding needed for a convolution.

    Args:
        kernel_size: The size of the kernel.
        dilation: The dilation of the convolution.
    """
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

    def __init__(self, period: 'int', kernel_size: 'int'=5, stride: 'int'=3, use_spectral_norm: 'bool'=False, checkpointing: 'bool'=False):
        super(DiscriminatorP, self).__init__()
        self.checkpointing = checkpointing
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        self.convs = torch.nn.ModuleList([norm_f(torch.nn.Conv2d(in_ch, out_ch, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))) for in_ch, out_ch in zip(in_channels, out_channels)])
        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = torch.nn.functional.pad(x, (0, n_pad), 'reflect')
        x = x.view(b, c, -1, self.period)
        for conv in self.convs:
            if self.training and self.checkpointing:
                x = checkpoint(conv, x, use_reentrant=False)
                x = checkpoint(self.lrelu, x, use_reentrant=False)
            else:
                x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

    def __init__(self, use_spectral_norm: 'bool'=False, checkpointing: 'bool'=False):
        super(DiscriminatorS, self).__init__()
        self.checkpointing = checkpointing
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList([norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)), norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)), norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)), norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)), norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)), norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            if self.training and self.checkpointing:
                x = checkpoint(conv, x, use_reentrant=False)
                x = checkpoint(self.lrelu, x, use_reentrant=False)
            else:
                x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(self, use_spectral_norm: 'bool'=False, checkpointing: 'bool'=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11, 17, 23, 37]
        self.checkpointing = checkpointing
        self.discriminators = torch.nn.ModuleList([DiscriminatorS(use_spectral_norm=use_spectral_norm, checkpointing=checkpointing)] + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm, checkpointing=checkpointing) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            if self.training and self.checkpointing:

                def forward_discriminator(d, y, y_hat):
                    y_d_r, fmap_r = d(y)
                    y_d_g, fmap_g = d(y_hat)
                    return y_d_r, fmap_r, y_d_g, fmap_g
                y_d_r, fmap_r, y_d_g, fmap_g = checkpoint(forward_discriminator, d, y, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ConvBlockRes(nn.Module):
    """
    A convolutional block with residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        momentum (float): Momentum for batch normalization.
    """

    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    """
    A residual encoder block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): Size of the average pooling kernel.
        n_blocks (int): Number of convolutional blocks in the block.
        momentum (float): Momentum for batch normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Encoder(nn.Module):
    """
    The encoder part of the DeepUnet.

    Args:
        in_channels (int): Number of input channels.
        in_size (int): Size of the input tensor.
        n_encoders (int): Number of encoder blocks.
        kernel_size (tuple): Size of the average pooling kernel.
        n_blocks (int): Number of convolutional blocks in each encoder block.
        out_channels (int): Number of output channels for the first encoder block.
        momentum (float): Momentum for batch normalization.
    """

    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: 'torch.Tensor'):
        concat_tensors: 'List[torch.Tensor]' = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            t, x = self.layers[i](x)
            concat_tensors.append(t)
        return x, concat_tensors


def sequence_mask(length: 'torch.Tensor', max_length: 'Optional[int]'=None):
    """
    Generate a sequence mask.

    Args:
        length: The lengths of the sequences.
        max_length: The maximum length of the sequences.
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class TextEncoder(torch.nn.Module):
    """
    Text Encoder with configurable embedding dimension.

    Args:
        out_channels (int): Output channels of the encoder.
        hidden_channels (int): Hidden channels of the encoder.
        filter_channels (int): Filter channels of the encoder.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        kernel_size (int): Kernel size of the convolutional layers.
        p_dropout (float): Dropout probability.
        embedding_dim (int): Embedding dimension for phone embeddings (v1 = 256, v2 = 768).
        f0 (bool, optional): Whether to use F0 embedding. Defaults to True.
    """

    def __init__(self, out_channels: 'int', hidden_channels: 'int', filter_channels: 'int', n_heads: 'int', n_layers: 'int', kernel_size: 'int', p_dropout: 'float', embedding_dim: 'int', f0: 'bool'=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone: 'torch.Tensor', pitch: 'Optional[torch.Tensor]', lengths: 'torch.Tensor'):
        x = self.emb_phone(phone)
        if pitch is not None and self.emb_pitch:
            x += self.emb_pitch(pitch)
        x *= math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = x.transpose(1, -1)
        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    Fused add tanh sigmoid multiply operation.

    Args:
        input_a: The first input tensor.
        input_b: The second input tensor.
        n_channels: The number of channels.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WaveNet(torch.nn.Module):
    """
    WaveNet residual blocks as used in WaveGlow.

    Args:
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation_rate (int): Dilation rate of the convolution.
        n_layers (int): Number of convolutional layers.
        gin_channels (int, optional): Number of conditioning channels. Defaults to 0.
        p_dropout (float, optional): Dropout probability. Defaults to 0.
    """

    def __init__(self, hidden_channels: 'int', kernel_size: 'int', dilation_rate, n_layers: 'int', gin_channels: 'int'=0, p_dropout: 'int'=0):
        super().__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd for proper padding.'
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.n_channels_tensor = torch.IntTensor([hidden_channels])
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)
        if gin_channels:
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name='weight')
        dilations = [(dilation_rate ** i) for i in range(n_layers)]
        paddings = [((kernel_size * d - d) // 2) for d in dilations]
        for i in range(n_layers):
            self.in_layers.append(torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilations[i], padding=paddings[i]), name='weight'))
            res_skip_channels = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            self.res_skip_layers.append(torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(hidden_channels, res_skip_channels, 1), name='weight'))

    def forward(self, x, x_mask, g=None):
        output = x.clone().zero_()
        g = self.cond_layer(g) if g is not None else None
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g[:, i * 2 * self.hidden_channels:(i + 1) * 2 * self.hidden_channels, :] if g is not None else 0
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class PosteriorEncoder(torch.nn.Module):
    """
    Posterior Encoder for inferring latent representation.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        hidden_channels (int): Number of hidden channels in the encoder.
        kernel_size (int): Kernel size of the convolutional layers.
        dilation_rate (int): Dilation rate of the convolutional layers.
        n_layers (int): Number of layers in the encoder.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 0.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', hidden_channels: 'int', kernel_size: 'int', dilation_rate: 'int', n_layers: 'int', gin_channels: 'int'=0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: 'torch.Tensor', x_lengths: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        z *= x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.enc._forward_pre_hooks.values():
            if hook.__module__ == 'torch.nn.utils.parametrizations.weight_norm' and hook.__class__.__name__ == 'WeightNorm':
                torch.nn.utils.remove_weight_norm(self.enc)
        return self


def apply_mask_(tensor: 'torch.Tensor', mask: 'Optional[torch.Tensor]'):
    return tensor.mul_(mask) if mask else tensor


def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation)))


def init_weights(m, mean=0.0, std=0.01):
    """
    Initialize the weights of a module.

    Args:
        m: The module to initialize.
        mean: The mean of the normal distribution.
        std: The standard deviation of the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock(torch.nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers with residual connections.
    """

    def __init__(self, channels: 'int', kernel_size: 'int'=3, dilations: 'Tuple[int]'=(1, 3, 5)):
        """
        Initializes the ResBlock.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel. Defaults to 3.
            dilations (Tuple[int]): Tuple of dilation rates for the convolution layers in the first set.
        """
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

    @staticmethod
    def _create_convs(channels: 'int', kernel_size: 'int', dilations: 'Tuple[int]'):
        """
        Creates a list of 1D convolutional layers with specified dilations.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel.
            dilations (Tuple[int]): Tuple of dilation rates for each convolution layer.
        """
        layers = torch.nn.ModuleList([create_conv1d_layer(channels, kernel_size, d) for d in dilations])
        layers.apply(init_weights)
        return layers

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor'=None):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x_residual = x
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = apply_mask_(x, x_mask)
            x = torch.nn.functional.leaky_relu_(conv1(x), LRELU_SLOPE)
            x = apply_mask_(x, x_mask)
            x = conv2(x)
            x += x_residual
        return apply_mask_(x, x_mask)

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)


class HiFiGANGenerator(torch.nn.Module):
    """
    HiFi-GAN Generator module for audio synthesis.

    This module implements the generator part of the HiFi-GAN architecture,
    which uses transposed convolutions for upsampling and residual blocks for
    refining the audio output. It can also incorporate global conditioning.

    Args:
        initial_channel (int): Number of input channels to the initial convolutional layer.
        resblock_kernel_sizes (list): List of kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): List of lists of dilation rates for the residual blocks, corresponding to each kernel size.
        upsample_rates (list): List of upsampling factors for each upsampling layer.
        upsample_initial_channel (int): Number of output channels from the initial convolutional layer, which is also the input to the first upsampling layer.
        upsample_kernel_sizes (list): List of kernel sizes for the transposed convolutional layers used for upsampling.
        gin_channels (int, optional): Number of input channels for the global conditioning. If 0, no global conditioning is used. Defaults to 0.
    """

    def __init__(self, initial_channel: 'int', resblock_kernel_sizes: 'list', resblock_dilation_sizes: 'list', upsample_rates: 'list', upsample_initial_channel: 'int', upsample_kernel_sizes: 'list', gin_channels: 'int'=0):
        super(HiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // 2 ** i, upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
            ch = upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None):
        x = self.conv_pre(x)
        if g is not None:
            x += self.cond(g)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu_(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu_(x)
        x = self.conv_post(x)
        x = torch.tanh_(x)
        return x

    def __prepare_scriptable__(self):
        for l in self.ups_and_resblocks:
            for hook in l._forward_pre_hooks.values():
                if hook.__module__ == 'torch.nn.utils.parametrizations.weight_norm' and hook.__class__.__name__ == 'WeightNorm':
                    torch.nn.utils.remove_weight_norm(l)
        return self

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SineGenerator(nn.Module):
    """
    Definition of sine generator

    Generates sine waveforms with optional harmonics and additive noise.
    Can be used to create harmonic noise source for neural vocoders.

    Args:
        samp_rate (int): Sampling rate in Hz.
        harmonic_num (int): Number of harmonic overtones (default 0).
        sine_amp (float): Amplitude of sine-waveform (default 0.1).
        noise_std (float): Standard deviation of Gaussian noise (default 0.003).
        voiced_threshold (float): F0 threshold for voiced/unvoiced classification (default 0).
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.merge = nn.Sequential(nn.Linear(self.dim, 1, bias=False), nn.Tanh())

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        rad_values = f0_values / self.sampling_rate % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :] < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        return sines

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        sine_waves = sine_waves - sine_waves.mean(dim=1, keepdim=True)
        return self.merge(sine_waves)


class MRFLayer(torch.nn.Module):
    """
    A single layer of the Multi-Receptive Field (MRF) block.

    This layer consists of two 1D convolutional layers with weight normalization
    and Leaky ReLU activation in between. The first convolution has a dilation,
    while the second has a dilation of 1. A skip connection is added from the input
    to the output.

    Args:
        channels (int): The number of input and output channels.
        kernel_size (int): The kernel size of the convolutional layers.
        dilation (int): The dilation rate for the first convolutional layer.
    """

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size * dilation - dilation) // 2, dilation=dilation))
        self.conv2 = weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, dilation=1))

    def forward(self, x: 'torch.Tensor'):
        y = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
        y = self.conv1(y)
        y = torch.nn.functional.leaky_relu_(y, LRELU_SLOPE)
        y = self.conv2(y)
        return x + y

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)


class MRFBlock(torch.nn.Module):
    """
    A Multi-Receptive Field (MRF) block.

    This block consists of multiple MRFLayers with different dilation rates.
    It applies each layer sequentially to the input.

    Args:
        channels (int): The number of input and output channels for the MRFLayers.
        kernel_size (int): The kernel size for the convolutional layers in the MRFLayers.
        dilations (list[int]): A list of dilation rates for the MRFLayers.
    """

    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for dilation in dilations:
            self.layers.append(MRFLayer(channels, kernel_size, dilation))

    def forward(self, x: 'torch.Tensor'):
        for layer in self.layers:
            x = layer(x)
        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()


class SourceModuleHnNSF(torch.nn.Module):
    """
    Source Module for generating harmonic and noise components for audio synthesis.

    This module generates a harmonic source signal using sine waves and adds
    optional noise. It's often used in neural vocoders as a source of excitation.

    Args:
        sample_rate (int): Sampling rate of the audio in Hz.
        harmonic_num (int, optional): Number of harmonic overtones to generate above the fundamental frequency (F0). Defaults to 0.
        sine_amp (float, optional): Amplitude of the sine wave components. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of the additive white Gaussian noise. Defaults to 0.003.
        voiced_threshod (float, optional): Threshold for the fundamental frequency (F0) to determine if a frame is voiced. If F0 is below this threshold, it's considered unvoiced. Defaults to 0.
    """

    def __init__(self, sample_rate: 'int', harmonic_num: 'int'=0, sine_amp: 'float'=0.1, add_noise_std: 'float'=0.003, voiced_threshod: 'float'=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGenerator(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: 'torch.Tensor', upsample_factor: 'int'=1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class HiFiGANMRFGenerator(torch.nn.Module):
    """
    HiFi-GAN generator with Multi-Receptive Field (MRF) blocks.

    This generator takes an input feature sequence and fundamental frequency (F0)
    as input and generates an audio waveform. It utilizes transposed convolutions
    for upsampling and MRF blocks for feature refinement. It can also condition
    on global conditioning features.

    Args:
        in_channel (int): Number of input channels.
        upsample_initial_channel (int): Number of channels after the initial convolution.
        upsample_rates (list[int]): List of upsampling rates for the transposed convolutions.
        upsample_kernel_sizes (list[int]): List of kernel sizes for the transposed convolutions.
        resblock_kernel_sizes (list[int]): List of kernel sizes for the convolutional layers in the MRF blocks.
        resblock_dilations (list[list[int]]): List of lists of dilation rates for the MRF blocks.
        gin_channels (int): Number of global conditioning input channels (0 if no global conditioning).
        sample_rate (int): Sampling rate of the audio.
        harmonic_num (int): Number of harmonics to generate.
        checkpointing (bool): Whether to use checkpointing to save memory during training (default: False).
    """

    def __init__(self, in_channel: 'int', upsample_initial_channel: 'int', upsample_rates: 'list[int]', upsample_kernel_sizes: 'list[int]', resblock_kernel_sizes: 'list[int]', resblock_dilations: 'list[list[int]]', gin_channels: 'int', sample_rate: 'int', harmonic_num: 'int', checkpointing: 'bool'=False):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.checkpointing = checkpointing
        self.f0_upsample = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)
        self.conv_pre = weight_norm(torch.nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3))
        self.upsamples = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()
        stride_f0s = [(math.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1) for i in range(len(upsample_rates))]
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2
            self.upsamples.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // 2 ** i, upsample_initial_channel // 2 ** (i + 1), kernel_size=k, stride=u, padding=padding, output_padding=u % 2)))
            """ handling odd upsampling rates
            #  s   k   p
            # 40  80  20
            # 32  64  16
            #  4   8   2
            #  2   3   1
            # 63 125  31
            #  9  17   4
            #  3   5   1
            #  1   1   0
            """
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2
            self.noise_convs.append(torch.nn.Conv1d(1, upsample_initial_channel // 2 ** (i + 1), kernel_size=kernel, stride=stride, padding=padding))
        self.mrfs = torch.nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // 2 ** (i + 1)
            self.mrfs.append(torch.nn.ModuleList([MRFBlock(channel, kernel_size=k, dilations=d) for k, d in zip(resblock_kernel_sizes, resblock_dilations)]))
        self.conv_post = weight_norm(torch.nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3))
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: 'torch.Tensor', f0: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None):
        f0 = self.f0_upsample(f0[:, None, :]).transpose(-1, -2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)
        x = self.conv_pre(x)
        if g is not None:
            x += self.cond(g)
        for ups, mrf, noise_conv in zip(self.upsamples, self.mrfs, self.noise_convs):
            x = torch.nn.functional.leaky_relu_(x, LRELU_SLOPE)
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
            else:
                x = ups(x)
            x += noise_conv(har_source)

            def mrf_sum(x, layers):
                return sum(layer(x) for layer in layers) / self.num_kernels
            if self.training and self.checkpointing:
                x = checkpoint(mrf_sum, x, mrf, use_reentrant=False)
            else:
                x = mrf_sum(x, mrf)
        x = torch.nn.functional.leaky_relu_(x)
        x = self.conv_post(x)
        x = torch.tanh_(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class HiFiGANNSFGenerator(torch.nn.Module):
    """
    Generator module based on the Neural Source Filter (NSF) architecture.

    This generator synthesizes audio by first generating a source excitation signal
    (harmonic and noise) and then filtering it through a series of upsampling and
    residual blocks. Global conditioning can be applied to influence the generation.

    Args:
        initial_channel (int): Number of input channels to the initial convolutional layer.
        resblock_kernel_sizes (list): List of kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): List of lists of dilation rates for the residual blocks, corresponding to each kernel size.
        upsample_rates (list): List of upsampling factors for each upsampling layer.
        upsample_initial_channel (int): Number of output channels from the initial convolutional layer, which is also the input to the first upsampling layer.
        upsample_kernel_sizes (list): List of kernel sizes for the transposed convolutional layers used for upsampling.
        gin_channels (int): Number of input channels for the global conditioning. If 0, no global conditioning is used.
        sr (int): Sampling rate of the audio.
        checkpointing (bool, optional): Whether to use gradient checkpointing to save memory during training. Defaults to False.
    """

    def __init__(self, initial_channel: 'int', resblock_kernel_sizes: 'list', resblock_dilation_sizes: 'list', upsample_rates: 'list', upsample_initial_channel: 'int', upsample_kernel_sizes: 'list', gin_channels: 'int', sr: 'int', checkpointing: 'bool'=False):
        super(HiFiGANNSFGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()
        channels = [(upsample_initial_channel // 2 ** (i + 1)) for i in range(len(upsample_rates))]
        stride_f0s = [(math.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1) for i in range(len(upsample_rates))]
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2
            self.ups.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // 2 ** i, channels[i], k, u, padding=padding, output_padding=u % 2)))
            """ handling odd upsampling rates
            #  s   k   p
            # 40  80  20
            # 32  64  16
            #  4   8   2
            #  2   3   1
            # 63 125  31
            #  9  17   4
            #  3   5   1
            #  1   1   0
            """
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2
            self.noise_convs.append(torch.nn.Conv1d(1, channels[i], kernel_size=kernel, stride=stride, padding=padding))
        self.resblocks = torch.nn.ModuleList([ResBlock(channels[i], k, d) for i in range(len(self.ups)) for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)])
        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x: 'torch.Tensor', f0: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None):
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x += self.cond(g)
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = torch.nn.functional.leaky_relu_(x, self.lrelu_slope)
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
            else:
                x = ups(x)
            x += noise_convs(har_source)

            def resblock_forward(x, blocks):
                return sum(block(x) for block in blocks) / len(blocks)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            if self.training and self.checkpointing:
                x = checkpoint(resblock_forward, x, blocks, use_reentrant=False)
            else:
                x = resblock_forward(x, blocks)
        x = torch.nn.functional.leaky_relu_(x)
        x = torch.tanh_(self.conv_post(x))
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if hook.__module__ == 'torch.nn.utils.parametrizations.weight_norm' and hook.__class__.__name__ == 'WeightNorm':
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if hook.__module__ == 'torch.nn.utils.parametrizations.weight_norm' and hook.__class__.__name__ == 'WeightNorm':
                    remove_weight_norm(l)
        return self


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer.

    This layer applies a scaling factor to the input based on a learnable weight.

    Args:
        channels (int): Number of input channels.
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation applied after scaling. Defaults to 0.2.
    """

    def __init__(self, *, channels: int, leaky_relu_slope: float=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.activation = nn.LeakyReLU(leaky_relu_slope, inplace=True)

    def forward(self, x: 'torch.Tensor'):
        gaussian = torch.randn_like(x) * self.weight[None, :, None]
        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    """
    Parallel residual block that applies multiple residual blocks with different kernel sizes in parallel.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_sizes (tuple[int], optional): Tuple of kernel sizes for the parallel residual blocks. Defaults to (3, 7, 11).
        dilation (tuple[int], optional): Tuple of dilation rates for the convolutional layers within the residual blocks. Defaults to (1, 3, 5).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(self, *, in_channels: int, out_channels: int, kernel_sizes: tuple[int]=(3, 7, 11), dilation: tuple[int]=(1, 3, 5), leaky_relu_slope: float=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3)
        self.blocks = nn.ModuleList([nn.Sequential(AdaIN(channels=out_channels), ResBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, leaky_relu_slope=leaky_relu_slope), AdaIN(channels=out_channels)) for kernel_size in kernel_sizes])

    def forward(self, x: 'torch.Tensor'):
        x = self.input_conv(x)
        results = [block(x) for block in self.blocks]
        return torch.mean(torch.stack(results), dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block[1].remove_parametrizations()


class RefineGANGenerator(nn.Module):
    """
    RefineGAN generator for audio synthesis.

    This generator uses a combination of downsampling, residual blocks, and parallel residual blocks
    to refine an input mel-spectrogram and fundamental frequency (F0) into an audio waveform.
    It can also incorporate global conditioning.

    Args:
        sample_rate (int, optional): Sampling rate of the audio. Defaults to 44100.
        downsample_rates (tuple[int], optional): Downsampling rates for the downsampling blocks. Defaults to (2, 2, 8, 8).
        upsample_rates (tuple[int], optional): Upsampling rates for the upsampling blocks. Defaults to (8, 8, 2, 2).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
        num_mels (int, optional): Number of mel-frequency bins in the input mel-spectrogram. Defaults to 128.
        start_channels (int, optional): Number of channels in the initial convolutional layer. Defaults to 16.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 256.
        checkpointing (bool, optional): Whether to use checkpointing for memory efficiency. Defaults to False.
    """

    def __init__(self, *, sample_rate: int=44100, downsample_rates: tuple[int]=(2, 2, 8, 8), upsample_rates: tuple[int]=(8, 8, 2, 2), leaky_relu_slope: float=0.2, num_mels: int=128, start_channels: int=16, gin_channels: int=256, checkpointing: bool=False, upsample_initial_channel=512):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing
        self.upp = np.prod(upsample_rates)
        self.m_source = SineGenerator(sample_rate)
        self.pre_conv = weight_norm(nn.Conv1d(in_channels=1, out_channels=upsample_initial_channel // 2, kernel_size=7, stride=1, padding=3, bias=False))
        stride_f0s = [(math.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1) for i in range(len(upsample_rates))]
        channels = upsample_initial_channel
        self.downsample_blocks = nn.ModuleList([])
        for i, u in enumerate(upsample_rates):
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2
            self.downsample_blocks.append(nn.Conv1d(in_channels=1, out_channels=channels // 2 ** (i + 2), kernel_size=kernel, stride=stride, padding=padding))
        self.mel_conv = weight_norm(nn.Conv1d(in_channels=num_mels, out_channels=channels // 2, kernel_size=7, stride=1, padding=3))
        if gin_channels != 0:
            self.cond = nn.Conv1d(256, channels // 2, 1)
        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])
        self.filters = nn.ModuleList([])
        for rate in upsample_rates:
            new_channels = channels // 2
            self.upsample_blocks.append(nn.Upsample(scale_factor=rate, mode='linear'))
            low_pass = nn.Conv1d(channels, channels, kernel_size=15, padding=7, groups=channels, bias=False)
            low_pass.weight.data.fill_(1.0 / 15)
            self.filters.append(low_pass)
            self.upsample_conv_blocks.append(ParallelResBlock(in_channels=channels + channels // 4, out_channels=new_channels, kernel_sizes=(3, 7, 11), dilation=(1, 3, 5), leaky_relu_slope=leaky_relu_slope))
            channels = new_channels
        self.conv_post = weight_norm(nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=7, stride=1, padding=3))

    def forward(self, mel: 'torch.Tensor', f0: 'torch.Tensor', g: 'torch.Tensor'=None):
        f0 = F.interpolate(f0.unsqueeze(1), size=mel.shape[-1] * self.upp, mode='linear')
        har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)
        x = self.pre_conv(har_source)
        x = F.interpolate(x, size=mel.shape[-1], mode='linear')
        mel = self.mel_conv(mel)
        if g is not None:
            mel += self.cond(g)
        x = torch.cat([mel, x], dim=1)
        for ups, res, down, flt in zip(self.upsample_blocks, self.upsample_conv_blocks, self.downsample_blocks, self.filters):
            x = F.leaky_relu_(x, self.leaky_relu_slope)
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = checkpoint(flt, x, use_reentrant=False)
                x = torch.cat([x, down(har_source)], dim=1)
                x = checkpoint(res, x, use_reentrant=False)
            else:
                x = ups(x)
                x = flt(x)
                x = torch.cat([x, down(har_source)], dim=1)
                x = res(x)
        x = F.leaky_relu_(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        x = torch.tanh_(x)
        return x

    def remove_parametrizations(self):
        remove_parametrizations(self.source_conv)
        remove_parametrizations(self.mel_conv)
        remove_parametrizations(self.conv_post)
        for block in self.downsample_blocks:
            block[1].remove_parametrizations()
        for block in self.upsample_conv_blocks:
            block.remove_parametrizations()


class LayerNorm(torch.nn.Module):
    """
    Layer normalization module.

    Args:
        channels (int): Number of channels.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, channels: 'int', eps: 'float'=1e-05):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class Flip(torch.nn.Module):
    """
    Flip module for flow-based models.

    This module flips the input along the time dimension.
    """

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ResidualCouplingLayer(torch.nn.Module):
    """
    Residual coupling layer for flow-based models.

    Args:
        channels (int): Number of channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation_rate (int): Dilation rate of the convolution.
        n_layers (int): Number of convolutional layers.
        p_dropout (float, optional): Dropout probability. Defaults to 0.
        gin_channels (int, optional): Number of conditioning channels. Defaults to 0.
        mean_only (bool, optional): Whether to use mean-only coupling. Defaults to False.
    """

    def __init__(self, channels: 'int', hidden_channels: 'int', kernel_size: 'int', dilation_rate: 'int', n_layers: 'int', p_dropout: 'float'=0, gin_channels: 'int'=0, mean_only: 'bool'=False):
        assert channels % 2 == 0, 'channels should be divisible by 2'
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = torch.nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None, reverse: 'bool'=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ResidualCouplingBlock(torch.nn.Module):
    """
    Residual Coupling Block for normalizing flow.

    Args:
        channels (int): Number of channels in the input.
        hidden_channels (int): Number of hidden channels in the coupling layer.
        kernel_size (int): Kernel size of the convolutional layers.
        dilation_rate (int): Dilation rate of the convolutional layers.
        n_layers (int): Number of layers in the coupling layer.
        n_flows (int, optional): Number of coupling layers in the block. Defaults to 4.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 0.
    """

    def __init__(self, channels: 'int', hidden_channels: 'int', kernel_size: 'int', dilation_rate: 'int', n_layers: 'int', n_flows: 'int'=4, gin_channels: 'int'=0):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = torch.nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None, reverse: 'bool'=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

    def __prepare_scriptable__(self):
        for i in range(self.n_flows):
            for hook in self.flows[i * 2]._forward_pre_hooks.values():
                if hook.__module__ == 'torch.nn.utils.parametrizations.weight_norm' and hook.__class__.__name__ == 'WeightNorm':
                    torch.nn.utils.remove_weight_norm(self.flows[i * 2])
        return self


def slice_segments(x: 'torch.Tensor', ids_str: 'torch.Tensor', segment_size: 'int'=4, dim: 'int'=2):
    """
    Slice segments from a tensor, handling tensors with different numbers of dimensions.

    Args:
        x (torch.Tensor): The tensor to slice.
        ids_str (torch.Tensor): The starting indices of the segments.
        segment_size (int, optional): The size of each segment. Defaults to 4.
        dim (int, optional): The dimension to slice across (2D or 3D tensors). Defaults to 2.
    """
    if dim == 2:
        ret = torch.zeros_like(x[:, :segment_size])
    elif dim == 3:
        ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size
        if dim == 2:
            ret[i] = x[i, idx_str:idx_end]
        else:
            ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    Randomly slice segments from a tensor.

    Args:
        x: The tensor to slice.
        x_lengths: The lengths of the sequences.
        segment_size: The size of each segment.
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = torch.rand([b], device=x.device) * ids_str_max
    ret = slice_segments(x, ids_str, segment_size, dim=3)
    return ret, ids_str


class Synthesizer(torch.nn.Module):
    """
    Base Synthesizer model.

    Args:
        spec_channels (int): Number of channels in the spectrogram.
        segment_size (int): Size of the audio segment.
        inter_channels (int): Number of channels in the intermediate layers.
        hidden_channels (int): Number of channels in the hidden layers.
        filter_channels (int): Number of channels in the filter layers.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of layers in the encoder.
        kernel_size (int): Size of the convolution kernel.
        p_dropout (float): Dropout probability.
        resblock (str): Type of residual block.
        resblock_kernel_sizes (list): Kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): Dilation sizes for the residual blocks.
        upsample_rates (list): Upsampling rates for the decoder.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes for the upsampling layers.
        spk_embed_dim (int): Dimension of the speaker embedding.
        gin_channels (int): Number of channels in the global conditioning vector.
        sr (int): Sampling rate of the audio.
        use_f0 (bool): Whether to use F0 information.
        text_enc_hidden_dim (int): Hidden dimension for the text encoder.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, spec_channels: 'int', segment_size: 'int', inter_channels: 'int', hidden_channels: 'int', filter_channels: 'int', n_heads: 'int', n_layers: 'int', kernel_size: 'int', p_dropout: 'float', resblock: 'str', resblock_kernel_sizes: 'list', resblock_dilation_sizes: 'list', upsample_rates: 'list', upsample_initial_channel: 'int', upsample_kernel_sizes: 'list', spk_embed_dim: 'int', gin_channels: 'int', sr: 'int', use_f0: 'bool', text_enc_hidden_dim: 'int'=768, vocoder: 'str'='HiFi-GAN', randomized: 'bool'=True, checkpointing: 'bool'=False, **kwargs):
        super().__init__()
        self.segment_size = segment_size
        self.use_f0 = use_f0
        self.randomized = randomized
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, text_enc_hidden_dim, f0=use_f0)
        None
        if use_f0:
            if vocoder == 'MRF HiFi-GAN':
                self.dec = HiFiGANMRFGenerator(in_channel=inter_channels, upsample_initial_channel=upsample_initial_channel, upsample_rates=upsample_rates, upsample_kernel_sizes=upsample_kernel_sizes, resblock_kernel_sizes=resblock_kernel_sizes, resblock_dilations=resblock_dilation_sizes, gin_channels=gin_channels, sample_rate=sr, harmonic_num=8, checkpointing=checkpointing)
            elif vocoder == 'RefineGAN':
                self.dec = RefineGANGenerator(sample_rate=sr, downsample_rates=upsample_rates[::-1], upsample_rates=upsample_rates, start_channels=16, num_mels=inter_channels, checkpointing=checkpointing)
            else:
                self.dec = HiFiGANNSFGenerator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, sr=sr, checkpointing=checkpointing)
        elif vocoder == 'MRF HiFi-GAN':
            None
            self.dec = None
        elif vocoder == 'RefineGAN':
            None
            self.dec = None
        else:
            self.dec = HiFiGANGenerator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, checkpointing=checkpointing)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    def _remove_weight_norm_from(self, module):
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, '__class__', None).__name__ == 'WeightNorm':
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(self, phone: 'torch.Tensor', phone_lengths: 'torch.Tensor', pitch: 'Optional[torch.Tensor]'=None, pitchf: 'Optional[torch.Tensor]'=None, y: 'Optional[torch.Tensor]'=None, y_lengths: 'Optional[torch.Tensor]'=None, ds: 'Optional[torch.Tensor]'=None):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            if self.randomized:
                z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
                if self.use_f0:
                    pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                    o = self.dec(z_slice, pitchf, g=g)
                else:
                    o = self.dec(z_slice, g=g)
                return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
            else:
                if self.use_f0:
                    o = self.dec(z, pitchf, g=g)
                else:
                    o = self.dec(z, g=g)
                return o, None, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        else:
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(self, phone: 'torch.Tensor', phone_lengths: 'torch.Tensor', pitch: 'Optional[torch.Tensor]'=None, nsff0: 'Optional[torch.Tensor]'=None, sid: 'torch.Tensor'=None, rate: 'Optional[torch.Tensor]'=None):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            rate (torch.Tensor, optional): Rate for time-stretching.
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g) if self.use_f0 else self.dec(z * x_mask, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class DepthWiseConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class GLU(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class Swish(nn.Module):

    def forward(self, x):
        return x * x.sigmoid()


class Transpose(nn.Module):

    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class ConformerConvModule(nn.Module):

    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.net = nn.Sequential(nn.LayerNorm(dim), Transpose((1, 2)), nn.Conv1d(dim, inner_dim * 2, 1), GLU(dim=1), DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding), Swish(), nn.Conv1d(inner_dim, dim, 1), Transpose((1, 2)), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t, (q, r))
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q[:remaining_rows])
    final_matrix = torch.cat(block_list)
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    return torch.diag(multiplier) @ final_matrix


def linear_attention(q, k, v):
    if v is None:
        out = torch.einsum('...ed,...nd->...ne', k, q)
        return out
    else:
        k_cumsum = k.sum(dim=-2)
        D_inv = 1.0 / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-08)
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=0.0001, device=None):
    b, h, *_ = data.shape
    data_normalizer = data.shape[-1] ** -0.25 if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)
    data_dash = torch.einsum('...id,...jd->...ij', data_normalizer * data, projection)
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = diag_data / 2.0 * data_normalizer ** 2
    diag_data = diag_data.unsqueeze(dim=-1)
    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * torch.exp(data_dash - diag_data + eps)
    return data_dash.type_as(data)


class FastAttention(nn.Module):

    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device
        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out


def empty(tensor):
    return tensor.numel() == 0


class SelfAttention(nn.Module):

    def __init__(self, dim, causal=False, heads=8, dim_head=64, local_heads=0, local_window_size=256, nb_features=None, feature_redraw_interval=1000, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, dropout=0.0, no_projection=False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal, generalized_attention=generalized_attention, kernel_fn=kernel_fn, qr_uniform_q=qr_uniform_q, no_projection=no_projection)
        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout, look_forward=int(not causal), rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(self, x, context=None, mask=None, context_mask=None, name=None, inference=False, **kwargs):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads
        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))
        attn_outs = []
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.0)
            if cross_attend:
                pass
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)
        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)
        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class _EncoderLayer(nn.Module):

    def __init__(self, parent: 'PCmer'):
        super().__init__()
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = SelfAttention(dim=parent.dim_model, heads=parent.num_heads, causal=False)

    def forward(self, phone, mask=None):
        phone = phone + self.attn(self.norm(phone), mask=mask)
        phone = phone + self.conformer(phone)
        return phone


class PCmer(nn.Module):

    def __init__(self, num_layers, num_heads, dim_model, dim_keys, dim_values, residual_dropout, attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        for layer in self._layers:
            phone = layer(phone, mask)
        return phone


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


class FCPE(nn.Module):

    def __init__(self, input_channel=128, out_dims=360, n_layers=12, n_chans=512, use_siren=False, use_full=False, loss_mse_scale=10, loss_l2_regularization=False, loss_l2_regularization_scale=1, loss_grad1_mse=False, loss_grad1_mse_scale=1, f0_max=1975.5, f0_min=32.7, confidence=False, threshold=0.05, use_input_conv=True):
        super().__init__()
        if use_siren is True:
            raise ValueError('Siren is not supported yet.')
        if use_full is True:
            raise ValueError('Full model is not supported yet.')
        self.loss_mse_scale = loss_mse_scale if loss_mse_scale is not None else 10
        self.loss_l2_regularization = loss_l2_regularization if loss_l2_regularization is not None else False
        self.loss_l2_regularization_scale = loss_l2_regularization_scale if loss_l2_regularization_scale is not None else 1
        self.loss_grad1_mse = loss_grad1_mse if loss_grad1_mse is not None else False
        self.loss_grad1_mse_scale = loss_grad1_mse_scale if loss_grad1_mse_scale is not None else 1
        self.f0_max = f0_max if f0_max is not None else 1975.5
        self.f0_min = f0_min if f0_min is not None else 32.7
        self.confidence = confidence if confidence is not None else False
        self.threshold = threshold if threshold is not None else 0.05
        self.use_input_conv = use_input_conv if use_input_conv is not None else True
        self.cent_table_b = torch.Tensor(np.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0], out_dims))
        self.register_buffer('cent_table', self.cent_table_b)
        _leaky = nn.LeakyReLU()
        self.stack = nn.Sequential(nn.Conv1d(input_channel, n_chans, 3, 1, 1), nn.GroupNorm(4, n_chans), _leaky, nn.Conv1d(n_chans, n_chans, 3, 1, 1))
        self.decoder = PCmer(num_layers=n_layers, num_heads=8, dim_model=n_chans, dim_keys=n_chans, dim_values=n_chans, residual_dropout=0.1, attention_dropout=0.1)
        self.norm = nn.LayerNorm(n_chans)
        self.n_out = out_dims
        self.dense_out = weight_norm(nn.Linear(n_chans, self.n_out))

    def forward(self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder='local_argmax'):
        if cdecoder == 'argmax':
            self.cdecoder = self.cents_decoder
        elif cdecoder == 'local_argmax':
            self.cdecoder = self.cents_local_decoder
        x = self.stack(mel.transpose(1, 2)).transpose(1, 2) if self.use_input_conv else mel
        x = self.decoder(x)
        x = self.norm(x)
        x = self.dense_out(x)
        x = torch.sigmoid(x)
        if not infer:
            gt_cent_f0 = self.f0_to_cent(gt_f0)
            gt_cent_f0 = self.gaussian_blurred_cent(gt_cent_f0)
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, gt_cent_f0)
            if self.loss_l2_regularization:
                loss_all = loss_all + l2_regularization(model=self, l2_alpha=self.loss_l2_regularization_scale)
            x = loss_all
        if infer:
            x = self.cdecoder(x)
            x = self.cent_to_f0(x)
            x = (1 + x / 700).log() if not return_hz_f0 else x
        return x

    def cents_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float('-INF')
            rtn = rtn * confident_mask
        return (rtn, confident) if self.confidence else rtn

    def cents_local_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0, 9) + (max_index - 4)
        local_argmax_index = torch.clamp(local_argmax_index, 0, self.n_out - 1)
        ci_l = torch.gather(ci, -1, local_argmax_index)
        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float('-INF')
            rtn = rtn * confident_mask
        return (rtn, confident) if self.confidence else rtn

    def cent_to_f0(self, cent):
        return 10.0 * 2 ** (cent / 1200.0)

    def f0_to_cent(self, f0):
        return 1200.0 * torch.log2(f0 / 10.0)

    def gaussian_blurred_cent(self, cents):
        mask = (cents > 0.1) & (cents < 1200.0 * np.log2(self.f0_max / 10.0))
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()


class Intermediate(nn.Module):
    """
    The intermediate layer of the DeepUnet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_inters (int): Number of convolutional blocks in the intermediate layer.
        n_blocks (int): Number of convolutional blocks in each intermediate block.
        momentum (float): Momentum for batch normalization.
    """

    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class ResDecoderBlock(nn.Module):
    """
    A residual decoder block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (tuple): Stride for transposed convolution.
        n_blocks (int): Number of convolutional blocks in the block.
        momentum (float): Momentum for batch normalization.
    """

    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), output_padding=out_padding, bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Decoder(nn.Module):
    """
    The decoder part of the DeepUnet.

    Args:
        in_channels (int): Number of input channels.
        n_decoders (int): Number of decoder blocks.
        stride (tuple): Stride for transposed convolution.
        n_blocks (int): Number of convolutional blocks in each decoder block.
        momentum (float): Momentum for batch normalization.
    """

    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    """
    The DeepUnet architecture.

    Args:
        kernel_size (tuple): Size of the average pooling kernel.
        n_blocks (int): Number of convolutional blocks in each encoder/decoder block.
        en_de_layers (int): Number of encoder/decoder layers.
        inter_layers (int): Number of convolutional blocks in the intermediate layer.
        in_channels (int): Number of input channels.
        en_out_channels (int): Number of output channels for the first encoder block.
    """

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):
    """
    A bidirectional GRU layer.

    Args:
        input_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        num_layers (int): Number of GRU layers.
    """

    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


N_CLASS = 360


N_MELS = 128


class E2E(nn.Module):
    """
    The end-to-end model.

    Args:
        n_blocks (int): Number of convolutional blocks in each encoder/decoder block.
        n_gru (int): Number of GRU layers.
        kernel_size (tuple): Size of the average pooling kernel.
        en_de_layers (int): Number of encoder/decoder layers.
        inter_layers (int): Number of convolutional blocks in the intermediate layer.
        in_channels (int): Number of input channels.
        en_out_channels (int): Number of output channels for the first encoder block.
    """

    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E, self).__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(BiGRU(3 * 128, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(torch.nn.Module):
    """
    Extracts Mel-spectrogram features from audio.

    Args:
        n_mel_channels (int): Number of Mel-frequency bands.
        sample_rate (int): Sampling rate of the audio.
        win_length (int): Length of the window function in samples.
        hop_length (int): Hop size between frames in samples.
        n_fft (int, optional): Length of the FFT window. Defaults to None, which uses win_length.
        mel_fmin (int, optional): Minimum frequency for the Mel filter bank. Defaults to 0.
        mel_fmax (int, optional): Maximum frequency for the Mel filter bank. Defaults to None.
        clamp (float, optional): Minimum value for clamping the Mel-spectrogram. Defaults to 1e-5.
    """

    def __init__(self, n_mel_channels, sample_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-05):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + '_' + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new)
        fft = torch.stft(audio, n_fft=n_fft_new, hop_length=hop_length_new, win_length=win_length_new, window=self.hann_window[keyshift_key], center=center, return_complex=True)
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


def compute_window_length(n_mels: 'int', sample_rate: 'int'):
    f_min = 0
    f_max = sample_rate / 2
    window_length_seconds = 8 * n_mels / (f_max - f_min)
    window_length = int(window_length_seconds * sample_rate)
    return 2 ** (window_length.bit_length() - 1)


class MultiScaleMelSpectrogramLoss(torch.nn.Module):

    def __init__(self, sample_rate: 'int'=24000, n_mels: 'list[int]'=[5, 10, 20, 40, 80, 160, 320, 480], loss_fn=torch.nn.L1Loss()):
        super().__init__()
        self.sample_rate = sample_rate
        self.loss_fn = loss_fn
        self.log_base = torch.log(torch.tensor(10.0))
        self.stft_params: 'list[tuple]' = []
        self.hann_window: 'dict[int, torch.Tensor]' = {}
        self.mel_banks: 'dict[int, torch.Tensor]' = {}
        self.stft_params = [(mel, compute_window_length(mel, sample_rate), self.sample_rate // 100) for mel in n_mels]

    def mel_spectrogram(self, wav: 'torch.Tensor', n_mels: 'int', window_length: 'int', hop_length: 'int'):
        dtype_device = str(wav.dtype) + '_' + str(wav.device)
        win_dtype_device = str(window_length) + '_' + dtype_device
        mel_dtype_device = str(n_mels) + '_' + dtype_device
        if win_dtype_device not in self.hann_window:
            self.hann_window[win_dtype_device] = torch.hann_window(window_length, device=wav.device, dtype=torch.float32)
        wav = wav.squeeze(1)
        stft = torch.stft(wav.float(), n_fft=window_length, hop_length=hop_length, window=self.hann_window[win_dtype_device], return_complex=True)
        magnitude = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-06)
        if mel_dtype_device not in self.mel_banks:
            self.mel_banks[mel_dtype_device] = torch.from_numpy(librosa_mel_fn(sr=self.sample_rate, n_mels=n_mels, n_fft=window_length, fmin=0, fmax=None))
        mel_spectrogram = torch.matmul(self.mel_banks[mel_dtype_device], magnitude)
        return mel_spectrogram

    def forward(self, real: 'torch.Tensor', fake: 'torch.Tensor'):
        loss = 0.0
        for p in self.stft_params:
            real_mels = self.mel_spectrogram(real, *p)
            fake_mels = self.mel_spectrogram(fake, *p)
            real_logmels = torch.log(real_mels.clamp(min=1e-05)) / self.log_base
            fake_logmels = torch.log(fake_mels.clamp(min=1e-05)) / self.log_base
            loss += self.loss_fn(real_logmels, fake_logmels)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaIN,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BiGRU,
     lambda: ([], {'input_features': 4, 'hidden_features': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ConformerConvModule,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConvBlockRes,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiscriminatorP,
     lambda: ([], {'period': 4}),
     lambda: ([torch.rand([4, 1, 4])], {}),
     False),
    (DiscriminatorS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (Encoder,
     lambda: ([], {'in_channels': 4, 'in_size': 4, 'n_encoders': 4, 'kernel_size': 4, 'n_blocks': 4}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     False),
    (FFN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'filter_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 2]), torch.rand([4, 4, 2])], {}),
     False),
    (Flip,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GLU,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Intermediate,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'n_inters': 4, 'n_blocks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MRFLayer,
     lambda: ([], {'channels': 4, 'kernel_size': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'channels': 4, 'out_channels': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MultiPeriodDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {}),
     False),
    (ResBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResEncoderBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transpose,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
]

class Test_IAHispano_Applio(_paritybench_base):
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

