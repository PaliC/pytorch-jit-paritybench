import sys
_module = sys.modules[__name__]
del sys
args = _module
data = _module
data_loader = _module
make_testdata = _module
evaluate = _module
exp = _module
exp_basic = _module
exp_main = _module
exp_multi = _module
exp_single = _module
AutoCorrelation = _module
Autoformer_EncDec = _module
Embed = _module
GAU_EncDec = _module
GateAttention_Family = _module
SelfAttention_Family = _module
Transformer_EncDec = _module
layers = _module
embed = _module
main = _module
AGCRN = _module
DSANet = _module
DeepAR = _module
GAU = _module
Gdnn = _module
LSTNet = _module
Lstm = _module
Mlp = _module
MultiHeadGAU = _module
TCN = _module
TPA = _module
Trans = _module
VanillaTransformer = _module
decoder = _module
encoder = _module
multiHeadAttention = _module
positionwiseFeedForward = _module
transformer = _module
utils = _module
models = _module
gdnn = _module
informer1 = _module
attn = _module
decoder = _module
embed = _module
encoder = _module
model = _module
lstm = _module
mlp = _module
Autoformer = _module
DeepAR = _module
EDGru = _module
EDGruAttention = _module
EDLstm = _module
Gaformer = _module
Informer = _module
MultiHeadGaformer = _module
Transformer = _module
seq2seq = _module
activation = _module
constants = _module
data = _module
loss = _module
masking = _module
metrics = _module
mylogger = _module
mylogging = _module
search = _module
timefeatures = _module
tools = _module
visualization = _module
plot_functions = _module

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


import numpy as np


import pandas as pd


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import warnings


import random


from typing import Optional


from torch import optim


import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


import inspect


from torch.utils.data import SubsetRandomSampler


import time


import matplotlib.pyplot as plt


import torch.nn.functional as F


import math


from math import sqrt


import torch.fft


from inspect import isfunction


from math import pi


from math import log


from collections import OrderedDict


import logging


from torch.autograd import Variable


from collections import namedtuple


from torch.nn.utils import weight_norm


from torch import nn


from typing import Union


from torch import Tensor


from torch import Generator


from typing import TypeVar


from typing import List


from typing import Tuple


from typing import Sequence


from torch import default_generator


from torch.utils.data import Subset


from torch import randperm


from torch.utils.data import Sampler


import re


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[..., i].unsqueeze(-1)
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :L - S, :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class Encoder(nn.Module):

    def __init__(self, enc_in, emb_dim, hid_dim, n_layers, embed, freq, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = DataEmbedding(enc_in, emb_dim, embed, freq, dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, x_enc, x_mark_enc):
        embedded = self.embedding(x_enc, x_mark_enc)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class DecoderLayer(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):

    def __init__(self, dec_in, emb_dim, hid_dim, n_layers, embed, freq, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = DataEmbedding(dec_in, emb_dim, embed, freq, dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, dec_in)

    def forward(self, input, input_mark, hidden, cell):
        embedded = self.embedding(input, input_mark)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell


class SpatialEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(SpatialEmbedding, self).__init__()
        self.spa_emb = nn.Embedding(c_in, d_model)

    def forward(self, x):
        x = x.long()
        spa_emb = self.spa_emb(x)
        return spa_emb


class DataEmbedding_wo_pos(nn.Module):

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_rotary_emb(freqs, t, start_index=0):
    freqs = freqs
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = t * freqs.cos() + rotate_half(t) * freqs.sin()
    return torch.cat((t_left, t, t_right), dim=-1)


def exists(val):
    return val is not None


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, skip=False, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1, learned_freq=False):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        self.cache = dict()
        self.skip = skip
        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def rotate_queries_or_keys(self, t, seq_dim=1):
        if self.skip:
            return t
        device = t.device
        seq_len = t.shape[seq_dim]
        freqs = self.forward(lambda : torch.arange(seq_len, device=device), cache_key=seq_len)
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]
        if isfunction(t):
            t = t()
        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        if exists(cache_key):
            self.cache[cache_key] = freqs
        return freqs


class ConvLayer(nn.Module):

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=padding, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderStack(nn.Module):

    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // 2 ** i_len
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        return x_stack, attns


class TriangularCausalMask:

    def __init__(self, B, L, gau=False, device='cpu'):
        if gau:
            mask_shape = [B, L, L]
        else:
            mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)

    @property
    def mask(self):
        return self._mask


def attention_normalize(a, dim=-1, method='softmax'):
    """不同的注意力归一化方案
    softmax：常规/标准的指数归一化；
    squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
    softmax_plus：来自 https://kexue.fm/archives/8823 。
    """
    if method == 'softmax':
        return torch.softmax(a, dim=dim)
    else:
        mask = (a > -torch.tensor(float('inf')) / 10).type(torch.float)
        l = torch.maximum(torch.sum(mask, dim=dim, keepdims=True), torch.tensor(1))
        if method == 'squared_relu':
            return torch.relu(a) ** 2 / l
        elif method == 'softmax_plus':
            return torch.softmax(a * torch.log(l) / np.log(512), dim=dim)
    return a


class GateAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(GateAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.activation = args.test_activation

    def forward(self, u, q, k, v, attn_mask):
        B, L, E = q.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum('bmd,bnd->bmn', q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, gau=True, device=q.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(attention_normalize(scale * scores, dim=-1, method=self.activation))
        V = u * torch.einsum('bmn,bnd->bmd', A, v)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class Gelu(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class Relu(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


def align(tensor, axes, ndim=None):
    """重新对齐tensor（批量版expand_dims）
    axes：原来的第i维对齐新tensor的第axes[i]维；
    ndim：新tensor的维度。
    """
    assert len(axes) == tensor.dim()
    assert ndim or min(axes) >= 0
    ndim = ndim or max(axes) + 1
    indices = [None] * ndim
    for i in axes:
        indices[i] = slice(None)
    return tensor[indices]


class ScaleOffset(nn.Module):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
         3、hidden_*系列参数仅为有条件输入时(conditional=True)使用，
            用于通过外部条件控制beta和gamma。
    """

    def __init__(self, key_size, scale=True, offset=True, conditional=False, hidden_units=None, hidden_activation='linear', hidden_initializer='glorot_uniform', **kwargs):
        super(ScaleOffset, self).__init__(**kwargs)
        self.key_size = key_size
        self.scale = scale
        self.offset = offset
        self.conditional = conditional
        self.hidden_units = hidden_units
        if self.offset is True:
            self.beta = nn.Parameter(torch.zeros(self.key_size))
        if self.scale is True:
            self.gamma = nn.Parameter(torch.ones(self.key_size))
        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Sequential(nn.Linear(self.hidden_units, self.hidden_units, bias=False), hidden_activation)
            if self.offset is not False and self.offset is not None:
                self.beta_dense = nn.Linear(self.key_size, self.key_size, bias=False)
                self.beta_dense.weight = nn.Parameter(torch.zeros(self.key_size, self.size))
            if self.scale is not False and self.scale is not None:
                self.gamma_dense = nn.Linear(self.key_size, self.key_size, bias=False)
                self.gamma_dense.weight = nn.Parameter(torch.zeros(self.key_size, self.size))

    def forward(self, inputs):
        """如果带有条件，则默认以list为输入，第二个是条件
        """
        if self.conditional:
            inputs, conds = inputs
            if self.hidden_units is not None:
                conds = self.hidden_dense(conds)
            conds = align(conds, [0, -1], inputs.dim())
        if self.scale is not False and self.scale is not None:
            gamma = self.gamma if self.scale is True else self.scale
            if self.conditional:
                gamma = gamma + self.gamma_dense(conds)
            inputs = inputs * gamma
        if self.offset is not False and self.offset is not None:
            beta = self.beta if self.offset is True else self.offset
            if self.conditional:
                beta = beta + self.beta_dense(conds)
            inputs = inputs + beta
        return inputs


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class GateAttentionLayer(nn.Module):

    def __init__(self, attention, d_model, uv_size, qk_size, activation='gelu', use_bias=True, use_conv=True, use_aff=True, **kwargs):
        super(GateAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.d_model = d_model
        self.uv_size = uv_size
        self.qk_size = qk_size
        self.use_bias = use_bias
        self.use_conv = use_conv
        self.use_aff = use_aff
        if activation == 'relu':
            self.activation = Relu()
        elif activation == 'gelu':
            self.activation = Gelu()
        else:
            self.activation = Swish()
        self.query_projection = nn.Sequential(nn.Linear(d_model, self.qk_size, self.use_bias), self.activation)
        self.key_projection = nn.Sequential(nn.Linear(d_model, self.qk_size, self.use_bias), self.activation)
        self.qk_projection = nn.Sequential(nn.Linear(d_model, self.qk_size, self.use_bias), self.activation)
        self.value_projection = nn.Sequential(nn.Linear(d_model, self.uv_size, self.use_bias), self.activation)
        self.rotary_emb = RotaryEmbedding(dim=self.qk_size, skip=False)
        if self.use_conv:
            self.o_dense = nn.Conv1d(in_channels=self.uv_size, out_channels=self.d_model, kernel_size=1)
            self.u_projection = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=self.uv_size, kernel_size=1), self.activation)
        else:
            self.o_dense = nn.Linear(self.uv_size, self.d_model, bias=self.use_bias)
            self.u_projection = nn.Sequential(nn.Linear(d_model, self.uv_size, self.use_bias), self.activation)
        self.q_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)

    def forward(self, u, queries, keys, values, attn_mask=None):
        if self.use_aff:
            qk = self.qk_projection(queries)
            q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)
        else:
            q = self.query_projection(queries)
            k = self.key_projection(keys)
        q, k = self.rotary_emb.rotate_queries_or_keys(q), self.rotary_emb.rotate_queries_or_keys(k)
        v = self.value_projection(values)
        if self.use_conv:
            u = self.u_projection(u.permute(0, 2, 1)).transpose(1, 2)
        else:
            u = self.u_projection(u)
        out, attn = self.inner_attention(u, q, k, v, attn_mask)
        if self.use_conv:
            o = self.o_dense(out.permute(0, 2, 1)).transpose(1, 2)
        else:
            o = self.o_dense(out)
        return o, attn


class MultiHeadGateAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(MultiHeadGateAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.activation = args.test_activation

    def forward(self, u, q, k, v, attn_mask):
        B, L, H, E = q.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, gau=True, device=q.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(attention_normalize(scale * scores, dim=-1, method=self.activation))
        V = u * torch.einsum('bhls,bshd->blhd', A, v)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class MultiHeadGateAttentionLayer(nn.Module):

    def __init__(self, attention, d_model, n_heads, uv_size, qk_size, activation='gelu', use_bias=True, use_conv=True, use_aff=True, **kwargs):
        super(MultiHeadGateAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.d_model = d_model
        self.n_heads = n_heads
        self.uv_size = uv_size
        self.qk_size = qk_size
        self.use_bias = use_bias
        self.use_conv = use_conv
        self.use_aff = use_aff
        if activation == 'relu':
            self.activation = Relu()
        elif activation == 'gelu':
            self.activation = Gelu()
        else:
            self.activation = Swish()
        self.query_projection = nn.Sequential(nn.Linear(d_model, self.qk_size * self.n_heads, self.use_bias), self.activation)
        self.key_projection = nn.Sequential(nn.Linear(d_model, self.qk_size * self.n_heads, self.use_bias), self.activation)
        self.qk_projection = nn.Sequential(nn.Linear(d_model, self.qk_size * self.n_heads, self.use_bias), self.activation)
        self.value_projection = nn.Sequential(nn.Linear(d_model, self.uv_size * self.n_heads, self.use_bias), self.activation)
        self.rotary_emb = RotaryEmbedding(dim=self.qk_size * self.n_heads, skip=False)
        if self.use_conv:
            self.u_projection = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=self.uv_size * self.n_heads, kernel_size=1), self.activation)
            self.o_dense = nn.Conv1d(in_channels=self.uv_size * self.n_heads, out_channels=self.d_model, kernel_size=1)
        else:
            self.u_projection = nn.Sequential(nn.Linear(d_model, self.uv_size * self.n_heads, self.use_bias), self.activation)
            self.o_dense = nn.Linear(self.uv_size * self.n_heads, self.d_model, bias=self.use_bias)
        self.q_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)

    def forward(self, u, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        if self.use_aff:
            qk = self.qk_projection(queries)
            q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)
        else:
            q = self.query_projection(queries)
            k = self.key_projection(keys)
        q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=1).view(B, L, H, -1)
        k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=1).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        if self.use_conv:
            u = self.u_projection(u.permute(0, 2, 1)).transpose(1, 2).view(B, L, H, -1)
        else:
            u = self.u_projection(u).view(B, L, H, -1)
        out, attn = self.inner_attention(u, q, k, v, attn_mask)
        out = out.view(B, L, -1)
        if self.use_conv:
            o = self.o_dense(out.permute(0, 2, 1)).transpose(1, 2)
        else:
            o = self.o_dense(out)
        return o, attn


class FullAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbMask:

    def __init__(self, B, H, L, index, scores, device='cpu'):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class AVWGCN(nn.Module):

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum('knm,bmc->bknc', supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv


class AGCRNCell(nn.Module):

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_size = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_size, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_size, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        state = state
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_size, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_size)


class AVWDCRNN(nn.Module):

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_size = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_size
        seq_length = x.shape[1]
        current_inputs = x
        output_attention = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_attention.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_attention

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class AGCRN(nn.Module):

    def __init__(self):
        super(AGCRN, self).__init__()
        self.num_nodes = args.input_size
        self.input_size = 1
        self.hidden_size = args.agcrn_hidden_size
        self.out_size = 1
        self.horizon = 1
        self.num_layers = args.agcrn_n_layers
        self.cheb_k = 2
        self.embed_dim = args.agcrn_embed_dim
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(self.num_nodes, self.input_size, self.hidden_size, self.cheb_k, self.embed_dim, self.num_layers)
        self.end_conv = nn.Conv2d(1, self.horizon * self.out_size, kernel_size=(1, self.hidden_size), bias=True)

    def forward(self, source):
        source = source.unsqueeze(-1)
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)
        output = output[:, -1:, :, :]
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.out_size, self.num_nodes)
        output = output.permute(0, 1, 3, 2)
        return output[:, 0, :, 0]


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


def generate_local_map_mask(chunk_size: 'int', attention_size: 'int', mask_future=False, device: 'torch.device'='cpu') ->torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)
    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size
    return torch.BoolTensor(local_map)


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self, d_model: 'int', q: 'int', v: 'int', h: 'int', attention_size: 'int'=None):
        """Initialize the Multi Head Block."""
        super().__init__()
        self._h = h
        self._attention_size = attention_size
        self._W_q = nn.Linear(d_model, q * self._h)
        self._W_k = nn.Linear(d_model, q * self._h)
        self._W_v = nn.Linear(d_model, v * self._h)
        self._W_o = nn.Linear(self._h * v, d_model)
        self._scores = None

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'Optional[str]'=None) ->torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(K)
        if self._attention_size is not None:
            attention_mask = generate_local_map_mask(K, self._attention_size, mask_future=False, device=self._scores.device)
            self._scores = self._scores.masked_fill(attention_mask, float('-inf'))
        if mask == 'subsequent':
            future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
            future_mask = future_mask
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))
        self._scores = F.softmax(self._scores, dim=-1)
        attention = torch.bmm(self._scores, values)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        self_attention = self._W_o(attention_heads)
        return self_attention

    @property
    def attention_map(self) ->torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError('Evaluate the model once to generate attention map')
        return self._scores


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.

    Apply two linear transformations to each input, separately but indetically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).

    Parameters
    ----------
    d_model:
        Dimension of input tensor.
    d_ff:
        Dimension of hidden layer, default is 2048.
    """

    def __init__(self, d_model: 'int', d_ff: 'Optional[int]'=2048):
        """Initialize the PFF block."""
        super().__init__()
        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Propagate forward the input through the PFF block.

        Apply the first linear transformation, then a relu actvation,
        and the second linear transformation.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        return self._linear2(F.relu(self._linear1(x)))


class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(self, window, n_multiv, n_kernels, w_kernel, d_k, d_v, d_model, d_inner, n_layers, n_head, drop_prob=0.1):
        """
        Args:

        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        """
        super(Single_Global_SelfAttn_Module, self).__init__()
        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob) for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)
        enc_slf_attn_list = []
        enout_sizeput = src_seq
        for enc_layer in self.layer_stack:
            enout_sizeput, enc_slf_attn = enc_layer(enout_sizeput)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enout_sizeput, enc_slf_attn_list
        enout_sizeput = self.out_linear(enout_sizeput)
        return enout_sizeput,


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(self, window, local, n_multiv, n_kernels, w_kernel, d_k, d_v, d_model, d_inner, n_layers, n_head, drop_prob=0.1):
        """
        Args:

        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        """
        super(Single_Local_SelfAttn_Module, self).__init__()
        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob) for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)
        enc_slf_attn_list = []
        enout_sizeput = src_seq
        for enc_layer in self.layer_stack:
            enout_sizeput, enc_slf_attn = enc_layer(enout_sizeput)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enout_sizeput, enc_slf_attn_list
        enout_sizeput = self.out_linear(enout_sizeput)
        return enout_sizeput,


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class DSANet(nn.Module):

    def __init__(self):
        super(DSANet, self).__init__()
        self.window = args.seq_len
        self.local = args.dsanet_local
        self.n_multiv = args.input_size
        self.n_kernels = args.dsanet_n_kernels
        self.w_kernel = args.dsanet_w_kernel
        self.d_model = args.dsanet_d_model
        self.d_inner = args.dsanet_d_inner
        self.n_layers = args.dsanet_n_layers
        self.n_head = args.dsanet_n_head
        self.d_k = args.dsanet_d_k
        self.d_v = args.dsanet_d_v
        self.drop_prob = args.dsanet_dropout
        self.__build_model()

    def __build_model(self):
        """
        Layout model
        """
        self.sgsf = Single_Global_SelfAttn_Module(window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels, w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)
        self.slsf = Single_Local_SelfAttn_Module(window=self.window, local=self.local, n_multiv=self.n_multiv, n_kernels=self.n_kernels, w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)
        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        """
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)
        sf_output = torch.transpose(sf_output, 1, 2)
        ar_output = self.ar(x)
        output = sf_output + ar_output
        return output[:, 0]


class DeepAR(nn.Module):

    def __init__(self, params):
        """
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        """
        super(DeepAR, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim)
        self.lstm = nn.LSTM(input_size=1 + params.cov_dim + params.embedding_dim, hidden_size=params.lstm_hidden_dim, num_layers=params.lstm_layers, bias=True, batch_first=False, dropout=params.lstm_dropout)
        """self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)"""
        for names in self.lstm._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)
        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_presigma = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x, idx, hidden, cell):
        """
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        """
        onehot_embed = self.embedding(idx)
        lstm_input = torch.cat((x, onehot_embed), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def test(self, x, v_batch, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.predict_steps, device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0), id_batch, decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    pred = gaussian.sample()
                    samples[j, :, t] = pred * v_batch[:, 0] + v_batch[:, 1]
                    if t < self.params.predict_steps - 1:
                        x[self.params.predict_start + t + 1, :, 0] = pred
            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma
        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0), id_batch, decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                if t < self.params.predict_steps - 1:
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma


class GAU(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        self.enc_embedding = DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.encoder = Encoder([EncoderLayer(GateAttentionLayer(GateAttention(False, attention_dropout=args.dropout, output_attention=args.output_attention), args.d_model, args.uv_size, args.qk_size, args.activation, args.use_bias, args.use_conv, args.use_aff), args.d_model, dropout=args.dropout) for l in range(args.e_layers)], [ConvLayer(args.d_model) for l in range(args.e_layers - 1)] if args.distil else None, norm_layer=torch.nn.LayerNorm(args.d_model))
        self.projection = nn.Linear(args.d_model, args.out_size, bias=True)

    def forward(self, x_enc, x_mark_enc=None, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        out = self.projection(enc_out[:, -1:, :])
        return out


class AU(nn.Module):

    def __init__(self, args):
        super(AU, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.encoder = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=args.output_attention), args.d_model, args.n_heads, mix=False), args.d_model, args.d_ff, dropout=args.dropout, activation=args.activation) for l in range(args.e_layers)], [ConvLayer(args.d_model) for l in range(args.e_layers - 1)] if args.distil else None, norm_layer=torch.nn.LayerNorm(args.d_model))
        self.projection = nn.Linear(args.d_model, args.out_size, bias=True)

    def forward(self, x_enc, x_mark_enc=None, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        out = self.projection(enc_out[:, -1:, :])
        return out


class Lstm(nn.Module):

    def __init__(self, args):
        super().__init__()
        enc_in, dec_in, emb_dim, hid_dim, n_layers, embed, freq, dropout = args.enc_in, args.dec_in, args.d_model, args.d_model, 3, args.embed, args.freq, args.dropout
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.pred_len = args.pred_len
        self.encoder = Encoder(enc_in, emb_dim, hid_dim, n_layers, embed, freq, dropout)
        self.decoder = Decoder(dec_in, emb_dim, hid_dim, n_layers, embed, freq, dropout)
        assert self.encoder.hid_dim == self.decoder.hid_dim, 'Hidden dimensions of encoder and decoder must be equal!'
        assert self.encoder.n_layers == self.decoder.n_layers, 'Encoder and decoder must have equal number of layers!'

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.training:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        else:
            teacher_forcing_ratio = 0
        batch_size, x_dec_len, dec_in = x_dec.shape
        outputs = torch.zeros(batch_size, x_dec_len - 1, dec_in)
        hidden, cell = self.encoder(x_enc, x_mark_enc)
        input = x_dec[:, 0, :].unsqueeze(dim=1)
        input_mark = x_mark_dec[:, 0, :].unsqueeze(dim=1)
        for t in range(1, x_dec_len):
            output, hidden, cell = self.decoder(input, input_mark, hidden, cell)
            outputs[:, t - 1, :] = output.squeeze(dim=1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = x_dec[:, t, :].unsqueeze(dim=1) if teacher_force else output
            input_mark = x_mark_dec[:, t, :].unsqueeze(dim=1)
        return outputs[:, -self.pred_len:, :]


class St_net(nn.Module):

    def __init__(self, n_spatial, gdnn_embed_size, embed_type, freq) ->None:
        super().__init__()
        self.spa_embed = SpatialEmbedding(n_spatial, gdnn_embed_size)
        self.tmp_embed = TemporalEmbedding(gdnn_embed_size, embed_type, freq)
        self.swish = Swish()

    def forward(self, x_temporal, x_spatial):
        x_spa_embed = self.spa_embed(x_spatial).squeeze()
        x_tmp_embed = self.tmp_embed(x_temporal)
        x_st = self.swish(x_spa_embed) * self.swish(x_tmp_embed)
        return x_st


class Gdnn(nn.Module):

    def __init__(self, args):
        """
        Args:
            n_spatial): num of spatial
            gdnn_embed_size (int): embedding dimension
            gdnn_hidden_size1 (int): lstm hidden dimension
            gdnn_out_size (int): lstm output dimension
            input_size (int): features dimension
            out_size (int): forescast dimension
            gdnn_hidden_size2 (int): combined net hidden dimension

        """
        super(Gdnn, self).__init__()
        n_spatial, gdnn_embed_size, embed_type, freq, input_size, gdnn_hidden_size1, gdnn_out_size, num_layers, gdnn_hidden_size2, out_size = args.n_spatial, args.gdnn_embed_size, args.embed, args.freq, args.input_size, args.gdnn_hidden_size1, args.gdnn_out_size, args.gdnn_n_layers, args.gdnn_hidden_size2, args.out_size
        self.st_net = St_net(n_spatial, gdnn_embed_size, embed_type, freq)
        self.lstm1 = Lstm(input_size, gdnn_hidden_size1, gdnn_out_size, num_layers)
        self.lstm2 = Lstm(gdnn_embed_size, gdnn_hidden_size1, gdnn_out_size, num_layers)
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(gdnn_out_size, gdnn_hidden_size2)
        self.linear2 = nn.Linear(gdnn_hidden_size2, out_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, x_temporal, x_spatial):
        x_st = self.st_net(x_temporal, x_spatial)
        lstm_out1 = self.swish(self.lstm1(x))
        lstm_out2 = self.sigmoid(self.lstm2(x_st))
        net_combine = self.dropout(lstm_out1 * lstm_out2)
        out1 = self.swish(self.linear1(net_combine))
        out = self.linear2(out1)
        return out


class LSTNet(nn.Module):

    def __init__(self, args):
        super(LSTNet, self).__init__()
        self.args = args
        self.P = args.seq_len
        self.m = args.input_size
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = args.highway_window
        self.conv1 = nn.Conv1d(self.m, self.hidC, self.Ck)
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.output = F.tanh

    def forward(self, x):
        if self.args.importance:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
        batch_size = x.size(0)
        c = x.transpose(1, 2)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view((batch_size, self.hidC, self.pt, self.skip))
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)
        res = self.linear1(r)
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
        res = res.unsqueeze(dim=1)
        return res


class BenchmarkLstm(nn.Module):
    """Example network for solving Oze datachallenge.

    Attributes
    ----------
    lstm: Torch LSTM
        LSTM layers.
    linear: Torch Linear
        Fully connected layer.
    """

    def __init__(self, args):
        """Defines LSTM and Linear layers.

        Parameters
        ----------
        input_size: int, optional
            Input dimension. Default is 45 (features_dim).
        hidden_size: int, optional
            Latent dimension. Default is 100.
        out_size: int, optional
            Output dimension. Default is 1.
        num_layers: int, optional
            Number of LSTM layers. Default is 3.
        """
        super().__init__()
        self.args = args
        input_size, hidden_size, out_size, num_layers = args.input_size, args.lstm_hidden_size, args.out_size, args.lstm_n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        """Propagate input through the network.

        Parameters
        ----------
        x: Tensor
            Input tensor with shape (batchsize, seq_len, features_dim)

        Returns
        -------
        output: Tensor
            Output tensor with shape (batchsize, *seq_len, out_size)
        """
        if self.args.importance:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
            x = x
        lstm_out, _ = self.lstm(x.float())
        lstm_out = self.dropout(lstm_out[:, -1:, :])
        output = self.linear(lstm_out)
        return output


class BenchmarkMlp(nn.Module):

    def __init__(self, args):
        super().__init__()
        input_size, hidden_size, out_size = args.input_size, args.mlp_hidden_size, args.out_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Sigmoid(), nn.Linear(hidden_size, out_size), nn.Sigmoid())

    def forward(self, x):
        outputs = self.net(x)
        return outputs


class MultiHeadGAU(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.encoder = Encoder([EncoderLayer(MultiHeadGateAttentionLayer(MultiHeadGateAttention(False, attention_dropout=args.dropout, output_attention=args.output_attention), args.d_model, args.n_heads, args.uv_size, args.qk_size, args.activation, args.use_bias, args.use_conv, args.use_aff), args.d_model, dropout=args.dropout) for l in range(args.e_layers)], [ConvLayer(args.d_model) for l in range(args.e_layers - 1)] if args.distil else None, norm_layer=torch.nn.LayerNorm(args.d_model))
        self.projection = nn.Linear(args.d_model, args.out_size, bias=True)

    def forward(self, x_enc, x_mark_enc=None, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        out = self.projection(enc_out[:, -1:, :])
        return out


class Chomp1d(nn.Module):
    """
    Args:
        remove padding
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):

    def __init__(self, args):
        super(TCN, self).__init__()
        self.args = args
        input_size, tcn_hidden_size, tcn_n_layers, tcn_dropout, out_size = args.input_size, args.tcn_hidden_size, args.tcn_n_layers, args.tcn_dropout, args.out_size
        num_channels = [tcn_hidden_size] * tcn_n_layers
        kernel_size = 2
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=tcn_dropout)]
        self.network = nn.Sequential(*layers)
        self.out_proj = nn.Linear(tcn_hidden_size, out_size)

    def forward(self, x):
        """
        Args:
            x: batch_size * seq_len, input_size
        """
        if self.args.importance:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
            x = x
        x = x.transpose(1, 2)
        out = self.network(x)[:, :, -1:]
        out = self.out_proj(out.transpose(1, 2))
        return out


class TPA_Attention(nn.Module):

    def __init__(self, seq_len, tpa_hidden_size):
        super(TPA_Attention, self).__init__()
        self.n_filters = 32
        self.filter_size = 1
        self.conv = nn.Conv2d(1, self.n_filters, (seq_len, self.filter_size))
        self.Wa = nn.Parameter(torch.rand(self.n_filters, tpa_hidden_size))
        self.Whv = nn.Linear(self.n_filters + tpa_hidden_size, tpa_hidden_size)

    def forward(self, hs, ht):
        """
        Args:
            ht: 最后一层，最后一步的hidden_state, B*hidden_size
            hs: 最后一层，每一步的lstm_out, B * seq_len *hidden_size

        """
        hs = hs.unsqueeze(1)
        H = self.conv(hs)[:, :, 0]
        H = H.transpose(1, 2)
        alpha = torch.sigmoid(torch.sum(H @ self.Wa * ht.unsqueeze(-1), dim=-1))
        V = torch.sum(H * alpha.unsqueeze(-1), dim=1)
        vh = torch.cat([V, ht], dim=1)
        return self.Whv(vh)


class TPA(nn.Module):

    def __init__(self, args):
        """
        Args:
            input_size: features dim
            tpa_ar_len: ar regression using last tpa_ar_len
            out_size: need to be predicted series, last out_size series
            default pred_len = 1 
        """
        super(TPA, self).__init__()
        input_size, seq_len, tpa_hidden_size, tpa_n_layers, tpa_ar_len, out_size = args.input_size, args.seq_len, args.tpa_hidden_size, args.tpa_n_layers, args.tpa_ar_len, args.out_size
        self.target_pos = args.target_pos
        self.ar_len = tpa_ar_len
        self.args = args
        self.input_proj = nn.Linear(input_size, tpa_hidden_size)
        self.lstm = nn.LSTM(input_size=tpa_hidden_size, hidden_size=tpa_hidden_size, num_layers=tpa_n_layers, batch_first=True)
        self.att = TPA_Attention(seq_len, tpa_hidden_size)
        self.out_proj = nn.Linear(tpa_hidden_size, out_size)
        pred_len = 1
        self.ar = nn.Linear(self.ar_len, pred_len)

    def forward(self, x):
        if self.args.importance:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
            x = x
        px = F.relu(self.input_proj(x))
        hs, (ht, _) = self.lstm(px)
        ht = ht[-1]
        final_h = self.att(hs, ht)
        ar_out = self.ar(x[:, -self.ar_len:, [self.target_pos]].transpose(1, 2))[:, :, 0]
        out = self.out_proj(final_h) + ar_out
        out = out.unsqueeze(1)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Trans(nn.Module):

    def __init__(self, args):
        super(Trans, self).__init__()
        input_size = args.input_size
        trans_hidden_size = args.trans_hidden_size
        trans_kernel_size = args.trans_kernel_size
        seq_len = args.seq_len
        n_trans_head = args.trans_n_heads
        trans_n_layers = args.trans_n_layers
        out_size = args.out_size
        self.conv = nn.Conv1d(input_size, trans_hidden_size, kernel_size=trans_kernel_size)
        self.pos_encoder = PositionalEncoding(trans_hidden_size, max_len=seq_len)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=trans_hidden_size, nhead=n_trans_head)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=trans_n_layers)
        self.fc = nn.Linear(trans_hidden_size, out_size)
        self.kernel_size = trans_kernel_size

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x).permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x).transpose(0, 1)[:, -1:]
        output = self.fc(x)
        return output


class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        self.encoder = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=args.output_attention), args.d_model, args.n_heads, mix=False), args.d_model, args.d_ff, dropout=args.dropout, activation=args.activation) for l in range(args.e_layers)], [ConvLayer(args.d_model) for l in range(args.e_layers - 1)] if args.distil else None, norm_layer=torch.nn.LayerNorm(args.d_model))
        self.decoder = Decoder([DecoderLayer(AttentionLayer(FullAttention(True, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, mix=args.mix), AttentionLayer(FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, mix=False), args.d_model, args.d_ff, dropout=args.dropout, activation=args.activation) for l in range(args.d_layers)], norm_layer=torch.nn.LayerNorm(args.d_model), projection=nn.Linear(args.d_model, args.out_size, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class MultiHeadAttentionChunk(MultiHeadAttention):
    """Multi Head Attention block with chunk.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks of constant size.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    chunk_size:
        Size of chunks to apply attention on. Last one may be smaller (see :class:`torch.Tensor.chunk`).
        Default is 168.
    """

    def __init__(self, d_model: 'int', q: 'int', v: 'int', h: 'int', attention_size: 'int'=None, chunk_size: 'Optional[int]'=168, **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)
        self._chunk_size = chunk_size
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._chunk_size, self._chunk_size)), diagonal=1).bool(), requires_grad=False)
        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._chunk_size, self._attention_size), requires_grad=False)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'Optional[str]'=None) ->torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]
        n_chunk = K // self._chunk_size
        queries = torch.cat(torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        keys = torch.cat(torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        values = torch.cat(torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._chunk_size)
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))
        if mask == 'subsequent':
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))
        self._scores = F.softmax(self._scores, dim=-1)
        attention = torch.bmm(self._scores, values)
        attention_heads = torch.cat(torch.cat(attention.chunk(n_chunk, dim=0), dim=1).chunk(self._h, dim=0), dim=-1)
        self_attention = self._W_o(attention_heads)
        return self_attention


class MultiHeadAttentionWindow(MultiHeadAttention):
    """Multi Head Attention block with moving window.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks using a moving window.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    window_size:
        Size of the window used to extract chunks.
        Default is 168
    padding:
        Padding around each window. Padding will be applied to input sequence.
        Default is 168 // 4 = 42.
    """

    def __init__(self, d_model: 'int', q: 'int', v: 'int', h: 'int', attention_size: 'int'=None, window_size: 'Optional[int]'=168, padding: 'Optional[int]'=168 // 4, **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)
        self._window_size = window_size
        self._padding = padding
        self._q = q
        self._v = v
        self._step = self._window_size - 2 * self._padding
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._window_size, self._window_size)), diagonal=1).bool(), requires_grad=False)
        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._window_size, self._attention_size), requires_grad=False)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'Optional[str]'=None) ->torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        batch_size = query.shape[0]
        query = F.pad(query.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        key = F.pad(key.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        value = F.pad(value.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)
        queries = queries.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        keys = keys.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        values = values.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._v, self._window_size)).transpose(1, 2)
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._window_size)
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))
        if mask == 'subsequent':
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))
        self._scores = F.softmax(self._scores, dim=-1)
        attention = torch.bmm(self._scores, values)
        attention = attention.reshape((batch_size * self._h, -1, self._window_size, self._v))
        attention = attention[:, :, self._padding:-self._padding, :]
        attention = attention.reshape((batch_size * self._h, -1, self._v))
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        self_attention = self._W_o(attention_heads)
        return self_attention


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """

    def __init__(self, args):
        super(Informer, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        Attn = ProbAttention if args.enc_attn == 'prob' else FullAttention
        self.encoder = Encoder([EncoderLayer(AttentionLayer(Attn(False, args.factor, attention_dropout=args.dropout, output_attention=args.output_attention), args.d_model, args.n_heads, mix=False), args.d_model, args.d_ff, dropout=args.dropout, activation=args.activation) for l in range(args.e_layers)], [ConvLayer(args.d_model) for l in range(args.e_layers - 1)] if args.distil else None, norm_layer=torch.nn.LayerNorm(args.d_model))
        self.decoder = Decoder([DecoderLayer(AttentionLayer(Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, mix=args.mix), AttentionLayer(FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, mix=False), args.d_model, args.d_ff, dropout=args.dropout, activation=args.activation) for l in range(args.d_layers)], norm_layer=torch.nn.LayerNorm(args.d_model), projection=nn.Linear(args.d_model, args.out_size, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class InformerStack(nn.Module):

    def __init__(self, enc_in, dec_in, out_size, out_len, factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512, dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', output_attention=False, distil=True, mix=True):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        inp_lens = list(range(len(e_layers)))
        encoders = [Encoder([EncoderLayer(AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads, mix=False), d_model, d_ff, dropout=dropout, activation=activation) for l in range(el)], [ConvLayer(d_model) for l in range(el - 1)] if distil else None, norm_layer=torch.nn.LayerNorm(d_model)) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        self.decoder = Decoder([DecoderLayer(AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads, mix=mix), AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads, mix=False), d_model, d_ff, dropout=dropout, activation=activation) for l in range(d_layers)], norm_layer=torch.nn.LayerNorm(d_model))
        self.projection = nn.Linear(d_model, out_size, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enout_size = self.enc_embedding(x_enc, x_mark_enc)
        enout_size, attns = self.encoder(enout_size, attn_mask=enc_self_mask)
        deout_size = self.dec_embedding(x_dec, x_mark_dec)
        deout_size = self.decoder(deout_size, enout_size, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        deout_size = self.projection(deout_size)
        if self.output_attention:
            return deout_size[:, -self.pred_len:, :], attns
        else:
            return deout_size[:, -self.pred_len:, :]


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, args):
        super(Autoformer, self).__init__()
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        kernel_size = args.moving_avg
        self.decomp = series_decomp(kernel_size)
        self.enc_embedding = DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        self.encoder = Encoder([EncoderLayer(AutoCorrelationLayer(AutoCorrelation(False, args.factor, attention_dropout=args.dropout, output_attention=args.output_attention), args.d_model, args.n_heads), args.d_model, args.d_ff, moving_avg=args.moving_avg, dropout=args.dropout, activation=args.activation) for l in range(args.e_layers)], norm_layer=my_Layernorm(args.d_model))
        self.decoder = Decoder([DecoderLayer(AutoCorrelationLayer(AutoCorrelation(True, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads), AutoCorrelationLayer(AutoCorrelation(False, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads), args.d_model, args.out_size, args.d_ff, moving_avg=args.moving_avg, dropout=args.dropout, activation=args.activation) for l in range(args.d_layers)], norm_layer=my_Layernorm(args.d_model), projection=nn.Linear(args.d_model, args.out_size, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        dec_out = trend_part + seasonal_part
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class Gru(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.pred_len = args.pred_len
        self.encoder = Encoder(args.enc_in, args.d_model, args.d_model, args.embed, args.freq, args.dropout)
        self.decoder = Decoder(args.dec_in, args.d_model, args.d_model, args.embed, args.freq, args.dropout)
        assert self.encoder.hid_dim == self.decoder.hid_dim, 'Hidden dimensions of encoder and decoder must be equal!'

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.training:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        else:
            teacher_forcing_ratio = 0
        batch_size, x_dec_len, dec_in = x_dec.shape
        outputs = torch.zeros(batch_size, x_dec_len - 1, dec_in)
        context = self.encoder(x_enc, x_mark_enc)
        hidden = context
        input = x_dec[:, 0, :].unsqueeze(dim=1)
        input_mark = x_mark_dec[:, 0, :].unsqueeze(dim=1)
        for t in range(1, x_dec_len):
            output, hidden = self.decoder(input, input_mark, hidden, context)
            outputs[:, t - 1, :] = output.squeeze(dim=1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = x_dec[:, t, :].unsqueeze(dim=1) if teacher_force else output
            input_mark = x_mark_dec[:, t, :].unsqueeze(dim=1)
        return outputs[:, -self.pred_len:, :]


class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size, x_enc_len = encoder_outputs.shape[0], encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, x_enc_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class GruAttention(nn.Module):

    def __init__(self, args):
        super().__init__()
        enc_in, dec_in, emb_dim, enc_hid_dim, dec_hid_dim, embed, freq, dropout = args.enc_in, args.dec_in, args.d_model, args.d_model, args.d_model, args.embed, args.freq, args.dropout
        self.pred_len = args.pred_len
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        attention = Attention(enc_hid_dim, dec_hid_dim)
        self.encoder = Encoder(enc_in, emb_dim, enc_hid_dim, dec_hid_dim, embed, freq, dropout)
        self.decoder = Decoder(dec_in, emb_dim, enc_hid_dim, dec_hid_dim, embed, freq, dropout, attention)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.training:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        else:
            teacher_forcing_ratio = 0
        batch_size, x_dec_len, dec_in = x_dec.shape
        outputs = torch.zeros(batch_size, x_dec_len - 1, dec_in)
        encoder_outputs, hidden = self.encoder(x_enc, x_mark_enc)
        input = x_dec[:, 0, :].unsqueeze(dim=1)
        input_mark = x_mark_dec[:, 0, :].unsqueeze(dim=1)
        for t in range(1, x_dec_len):
            output, hidden = self.decoder(input, input_mark, hidden, encoder_outputs)
            outputs[:, t - 1, :] = output.squeeze(dim=1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = x_dec[:, t, :].unsqueeze(dim=1) if teacher_force else output
            input_mark = x_mark_dec[:, t, :].unsqueeze(dim=1)
        return outputs[:, -self.pred_len:, :]


class Gaformer(nn.Module):
    """GAU-α
    改动：基本模块换成GAU
    链接：https://kexue.fm/archives/9052
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.encoder = Encoder([EncoderLayer(GateAttentionLayer(GateAttention(False, attention_dropout=args.dropout, output_attention=args.output_attention), args.d_model, args.uv_size, args.qk_size, args.activation, args.use_bias, args.use_conv, args.use_aff), args.d_model, dropout=args.dropout) for l in range(args.e_layers)], [ConvLayer(args.d_model) for l in range(args.e_layers - 1)] if args.distil else None, norm_layer=torch.nn.LayerNorm(args.d_model))
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        if args.dec_selfattn == 'gate':
            selfattnlayer = GateAttentionLayer(GateAttention(True, attention_dropout=args.dropout, output_attention=False), args.d_model, args.uv_size, args.qk_size, args.activation, args.use_bias, args.use_conv, args.use_aff)
        else:
            Attn = ProbAttention if args.dec_selfattn == 'prob' else FullAttention
            selfattnlayer = AttentionLayer(Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, mix=args.mix)
        if args.dec_crossattn == 'gate':
            """gateAttenLayer 包含FNN, 因此用GAU Decoderlayer 剔除FNN"""
            crossattnlayer = GateAttentionLayer(GateAttention(False, attention_dropout=args.dropout, output_attention=False), args.d_model, args.uv_size, args.qk_size, args.activation, args.use_bias, args.use_conv, args.use_aff)
            decoderlayer = DecoderLayer1(selfattnlayer, crossattnlayer, args.d_model, dropout=args.dropout)
        else:
            """不包含FNN，因此用Transformer Decoderlayer"""
            Attn = ProbAttention if args.dec_crossattn == 'prob' else FullAttention
            crossattnlayer = AttentionLayer(Attn(False, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, mix=False)
            decoderlayer = DecoderLayer2(selfattnlayer, crossattnlayer, args.d_model, args.d_ff, dropout=args.dropout, activation=args.activation)
        self.decoder = Decoder([decoderlayer for l in range(args.d_layers)], norm_layer=torch.nn.LayerNorm(args.d_model), projection=nn.Linear(args.d_model, args.out_size, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class MultiHeadGaformer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        self.enc_embedding = DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.encoder = Encoder([EncoderLayer(MultiHeadGateAttentionLayer(MultiHeadGateAttention(False, attention_dropout=args.dropout, output_attention=args.output_attention), args.d_model, args.n_heads, args.uv_size, args.qk_size, args.activation, args.use_bias, args.use_conv, args.use_aff), args.d_model, dropout=args.dropout) for l in range(args.e_layers)], [ConvLayer(args.d_model) for l in range(args.e_layers - 1)] if args.distil else None, norm_layer=torch.nn.LayerNorm(args.d_model))
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        if args.dec_selfattn == 'gate':
            selfattnlayer = MultiHeadGateAttentionLayer(MultiHeadGateAttention(True, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, args.uv_size, args.qk_size, args.activation, args.use_bias, args.use_conv, args.use_aff)
            self.dec_embedding = DataEmbedding_wo_pos(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        else:
            Attn = ProbAttention if args.dec_selfattn == 'prob' else FullAttention
            selfattnlayer = AttentionLayer(Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, mix=args.mix)
        if args.dec_crossattn == 'gate':
            """gateAttenLayer 包含FNN, 因此用GAU Decoderlayer 剔除FNN"""
            crossattnlayer = MultiHeadGateAttentionLayer(MultiHeadGateAttention(False, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, args.uv_size, args.qk_size, args.activation, args.use_bias, args.use_conv, args.use_aff)
            decoderlayer = DecoderLayer1(selfattnlayer, crossattnlayer, args.d_model, dropout=args.dropout)
        else:
            """不包含FNN，因此用Transformer Decoderlayer"""
            Attn = ProbAttention if args.dec_crossattn == 'prob' else FullAttention
            crossattnlayer = AttentionLayer(Attn(False, args.factor, attention_dropout=args.dropout, output_attention=False), args.d_model, args.n_heads, mix=False)
            decoderlayer = DecoderLayer2(selfattnlayer, crossattnlayer, args.d_model, args.d_ff, dropout=args.dropout, activation=args.activation)
        self.decoder = Decoder([decoderlayer for l in range(args.d_layers)], norm_layer=torch.nn.LayerNorm(args.d_model), projection=nn.Linear(args.d_model, args.out_size, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class OZELoss(nn.Module):
    """Custom loss for TRNSys metamodel.

    Compute, for temperature and consumptions, the intergral of the squared differences
    over time. Sum the log with a coeficient ``alpha``.

    .. math::
        \\Delta_T = \\sqrt{\\int (y_{est}^T - y^T)^2}

        \\Delta_Q = \\sqrt{\\int (y_{est}^Q - y^Q)^2}

        loss = log(1 + \\Delta_T) + \\alpha \\cdot log(1 + \\Delta_Q)

    Parameters:
    -----------
    alpha:
        Coefficient for consumption. Default is ``0.3``.
    """

    def __init__(self, reduction: 'str'='mean', alpha: 'float'=0.3):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.base_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, y_true: 'torch.Tensor', y_pred: 'torch.Tensor') ->torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Parameters
        ----------
        y_true:
            Target value.
        y_pred:
            Estimated value.

        Returns
        -------
        Loss as a tensor with gradient attached.
        """
        delta_Q = self.base_loss(y_pred[..., :-1], y_true[..., :-1])
        delta_T = self.base_loss(y_pred[..., -1], y_true[..., -1])
        if self.reduction == 'none':
            delta_Q = delta_Q.mean(dim=(1, 2))
            delta_T = delta_T.mean(dim=1)
        return torch.log(1 + delta_T) + self.alpha * torch.log(1 + delta_Q)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AR,
     lambda: ([], {'window': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AutoCorrelation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BenchmarkLstm,
     lambda: ([], {'args': _mock_config(input_size=4, lstm_hidden_size=4, out_size=4, lstm_n_layers=1, importance=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BenchmarkMlp,
     lambda: ([], {'args': _mock_config(input_size=4, mlp_hidden_size=4, out_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLayer,
     lambda: ([], {'c_in': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DataEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (DataEmbedding_wo_pos,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'enc_in': 4, 'emb_dim': 4, 'hid_dim': 4, 'n_layers': 1, 'embed': 4, 'freq': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (Gelu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lstm,
     lambda: ([], {'args': _mock_config(enc_in=4, dec_in=4, d_model=4, embed=4, freq=4, dropout=0.5, teacher_forcing_ratio=4, pred_len=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'d_model': 4, 'q': 4, 'v': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (OZELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ProbAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Relu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleOffset,
     lambda: ([], {'key_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SpatialEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (St_net,
     lambda: ([], {'n_spatial': 4, 'gdnn_embed_size': 4, 'embed_type': 4, 'freq': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TCN,
     lambda: ([], {'args': _mock_config(input_size=4, tcn_hidden_size=4, tcn_n_layers=1, tcn_dropout=0.5, out_size=4, importance=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TemporalEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TimeFeatureEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TokenEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Trans,
     lambda: ([], {'args': _mock_config(input_size=4, trans_hidden_size=4, trans_kernel_size=4, seq_len=4, trans_n_heads=4, trans_n_layers=1, out_size=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (moving_avg,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (my_Layernorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (series_decomp,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 2, 4])], {}),
     True),
]

class Test_hyliush_deep_time_series(_paritybench_base):
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

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

