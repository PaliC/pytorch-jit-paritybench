import sys
_module = sys.modules[__name__]
del sys
api = _module
app = _module
autofill = _module
general = _module
generate = _module
hardware = _module
models = _module
outputs = _module
settings = _module
static = _module
test = _module
ws = _module
websockets = _module
data = _module
manager = _module
notification = _module
bot = _module
bot_model_manager = _module
config = _module
core = _module
helper = _module
image2image = _module
listeners = _module
shared = _module
txt2img = _module
compile = _module
common = _module
clip = _module
unet = _module
vae = _module
attention = _module
clip = _module
embeddings = _module
mapping = _module
resnet = _module
unet_2d_condition = _module
unet_blocks = _module
vae = _module
_config = _module
api_settings = _module
bot_settings = _module
default_settings = _module
flags_settings = _module
frontend_settings = _module
interrogator_settings = _module
kdiffusion_sampler_config = _module
sampler_config = _module
errors = _module
cloudflare_r2 = _module
files = _module
flags = _module
functions = _module
gpu = _module
adetailer = _module
ait = _module
aitemplate = _module
pipeline = _module
base_model = _module
esrgan = _module
real_esrgan = _module
upscale = _module
RRDB = _module
SPSR = _module
SRVGG = _module
__index__ = _module
block = _module
dataops = _module
net_interp = _module
functions = _module
injectables = _module
lora = _module
lycoris = _module
textual_inversion = _module
utils = _module
onnx = _module
pipeline = _module
pytorch = _module
pipeline = _module
pytorch = _module
sdxl = _module
pipeline = _module
sdxl = _module
utilities = _module
anisotropic = _module
cfg = _module
controlnet = _module
kohya_hires = _module
latents = _module
lwp = _module
philox = _module
prompt_expansion = _module
downloader = _module
expand = _module
random = _module
sag = _module
cross_attn = _module
diffusers = _module
kdiff = _module
sag_utils = _module
scalecrafter = _module
scheduling = _module
vae = _module
inference_callbacks = _module
install_requirements = _module
base_interrogator = _module
clip = _module
deepdanbooru = _module
flamingo = _module
deepdanbooru_model = _module
websocket_logging = _module
optimizations = _module
attn = _module
flash_attention = _module
multihead_attention = _module
sub_quadratic = _module
autocast_utils = _module
stable_fast = _module
trace_utils = _module
context_manager = _module
dtype = _module
hypertile = _module
offload = _module
pytorch_optimizations = _module
upcast = _module
png_metadata = _module
queue = _module
k_adapter = _module
unipc_adapter = _module
dpmpp_2m = _module
heunpp = _module
lcm = _module
restart = _module
sasolver = _module
denoiser = _module
hijack = _module
scheduling = _module
sigmas = _module
types = _module
unipc = _module
noise_scheduler = _module
unipc = _module
utility = _module
shared_dependent = _module
thread = _module
types = _module
main = _module
convert_diffusers_to_sd = _module
parse_safetensors_keys = _module
const = _module
test_ait = _module
test_pytorch = _module
test_pytorch_sdxl = _module
test_main = _module
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


from typing import List


import torch


import logging


from typing import Literal


from typing import Optional


import time


from typing import Tuple


from typing import Union


from inspect import isfunction


from typing import Any


import math


from typing import Dict


from typing import TYPE_CHECKING


import inspect


from typing import Callable


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


from enum import Enum


import numpy as np


import functools


import re


import torch.nn as nn


import torch.nn.functional as F


from functools import partial


from typing import Sized


from copy import deepcopy


from typing import Type


import warnings


from time import time


from typing import TypeVar


from numpy.random import MT19937


from numpy.random import RandomState


from numpy.random import SeedSequence


from torch.onnx import export


from torch import Generator as native


import scipy


from torch.ao.quantization import get_default_qconfig_mapping


from torch.ao.quantization.backend_config.tensorrt import get_tensorrt_backend_config_dict


from torch.ao.quantization.quantize_fx import convert_fx


from torch.ao.quantization.quantize_fx import convert_to_reference_fx


from torch.ao.quantization.quantize_fx import prepare_fx


from typing import NamedTuple


from typing import Protocol


from logging import getLogger


from typing import Coroutine


from uuid import uuid4


class AttentionBlock(nn.Module):

    def __init__(self, batch_size: 'int', height: 'int', width: 'int', channels: 'int', num_head_channels: 'Optional[int]'=None, num_groups: 'int'=32, rescale_output_factor: 'float'=1.0, eps: 'float'=1e-05, dtype='float16') ->None:
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_groups, channels, eps, dtype=dtype)
        self.attention = nn.CrossAttention(channels, height * width, height * width, self.num_heads, qkv_bias=True, dtype=dtype)
        self.rescale_output_factor = rescale_output_factor

    def forward(self, hidden_states: 'Tensor') ->Tensor:
        residual = hidden_states
        hidden_states = self.group_norm(hidden_states)
        o_shape = hidden_states.shape()
        batch_dim = o_shape[0]
        hidden_states = reshape()(hidden_states, [batch_dim, -1, self.channels])
        res = self.attention(hidden_states, hidden_states, hidden_states, residual) * (1 / self.rescale_output_factor)
        res = reshape()(res, o_shape)
        return res


def default(val, d) ->Any:
    if val is not None:
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):

    def __init__(self, query_dim: 'int', context_dim: 'Optional[int]'=None, heads: 'int'=8, dim_head: 'int'=64, dropout: 'float'=0.0, dtype: 'str'='float16') ->None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale: 'float' = dim_head * -0.5
        self.heads: 'int' = heads
        self.dim_head: 'int' = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False, dtype=dtype)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim, dtype=dtype), nn.Dropout(dropout, dtype=dtype))

    def forward(self, x: 'Tensor', context: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None, residual: 'Optional[Tensor]'=None) ->Tensor:
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        bs = q.shape()[0]
        q = ops.reshape()(q, [bs, -1, self.heads, self.dim_head])
        k = ops.reshape()(k, [bs, -1, self.heads, self.dim_head])
        v = ops.reshape()(v, [bs, -1, self.heads, self.dim_head])
        q = ops.permute()(q, [0, 2, 1, 3])
        k = ops.permute()(k, [0, 2, 1, 3])
        v = ops.permute()(v, [0, 2, 1, 3])
        attn_op = ops.mem_eff_attention(causal=False)
        if not USE_CUDA:
            attn_op = ops.bmm_softmax_bmm_permute(shape=(self.heads,), scale=self.scale)
        out = attn_op(ops.reshape()(q, [bs, self.heads, -1, self.dim_head]), ops.reshape()(k, [bs, self.heads, -1, self.dim_head]), ops.reshape()(v, [bs, self.heads, -1, self.dim_head]))
        out = ops.reshape()(out, [bs, -1, self.heads * self.dim_head])
        proj = self.to_out(out)
        proj = ops.reshape()(proj, [bs, -1, self.heads * self.dim_head])
        if residual is not None:
            return proj + residual
        return proj


class GEGLU(nn.Module):

    def __init__(self, dim_in: 'int', dim_out: 'int', dtype: 'str'='float16') ->None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, specialization='mul', dtype=dtype)
        self.gate = nn.Linear(dim_in, dim_out, specialization='fast_gelu', dtype=dtype)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.proj(x, self.gate(x))


class FeedForward(nn.Module):

    def __init__(self, dim: 'int', dim_out: 'Optional[int]'=None, mult: 'int'=4, glu: 'bool'=False, dropout: 'float'=0.0, dtype: 'str'='float16') ->None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim, specialization='fast_gelu', dtype=dtype)) if not glu else GEGLU(dim, inner_dim, dtype=dtype)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout, dtype=dtype), nn.Linear(inner_dim, dim_out, dtype=dtype))

    def forward(self, x: 'Tensor', residual: 'Optional[Tensor]'=None) ->Tensor:
        shape = ops.size()(x)
        x = self.net(x)
        x = ops.reshape()(x, shape)
        if residual is not None:
            return x + residual
        return x


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim: 'int', n_heads: 'int', d_head: 'int', dropout: 'float'=0.0, context_dim: 'Optional[int]'=None, gated_ff: 'bool'=True, checkpoint: 'bool'=True, only_cross_attention: 'bool'=False, dtype: 'str'='float16') ->None:
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.attn1 = CrossAttention(query_dim=dim, context_dim=context_dim if only_cross_attention else None, heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)
        if context_dim is not None:
            self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype)
        else:
            self.attn2 = None
        self.norm1 = nn.LayerNorm(dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype)
        self.norm3 = nn.LayerNorm(dim, dtype=dtype)
        self.checkpoint = checkpoint
        self.param = dim, n_heads, d_head, context_dim, gated_ff, checkpoint

    def forward(self, x: 'Tensor', context: 'Optional[Tensor]'=None) ->Tensor:
        x = self.attn1(self.norm1(x), residual=x, context=context if self.only_cross_attention else None)
        if self.attn1 is not None:
            x = self.attn2(self.norm2(x), context=context, residual=x)
        x = self.ff(self.norm3(x), residual=x)
        return x


def Normalize(in_channels: 'int', dtype: 'str'='float16'):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-06, affine=True, dtype=dtype)


class SpatialTransformer(nn.Module):

    def __init__(self, in_channels: 'int', n_heads: 'int', d_head: 'int', depth: 'int'=1, dropout: 'float'=0.0, context_dim: 'Optional[int]'=None, use_linear_projection: 'bool'=False, only_cross_attention: 'bool'=False, dtype: 'str'='float16') ->None:
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels, dtype=dtype)
        self.use_linear_projection = use_linear_projection
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim, dtype=dtype)
        else:
            self.proj_in = nn.Conv2dBias(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, only_cross_attention=only_cross_attention, dtype=dtype) for d in range(depth)])
        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels, dtype=dtype)
        else:
            self.proj_out = nn.Conv2dBias(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype)

    def forward(self, x: 'Tensor', context: 'Optional[Tensor]'=None) ->Tensor:
        b, h, w, c = x.shape()
        x_in = x
        x = self.norm(x)
        if self.use_linear_projection:
            x = ops.reshape()(x, [b, -1, c])
            x = self.proj_in(x)
        else:
            x = self.proj_in(x)
            x = ops.reshape()(x, [b, -1, c])
        for block in self.transformer_blocks:
            x = block(x, context=context)
        if self.use_linear_projection:
            x = self.proj_out(x)
            x = ops.reshape()(x, [b, h, w, c])
        else:
            x = ops.reshape()(x, [b, h, w, c])
            x = self.proj_out(x)
        return x + x_in


class CLIPAttention(nn.Module):

    def __init__(self, hidden_size: 'int'=768, num_attention_heads: 'int'=12, attention_dropout: 'float'=0.0, batch_size: 'int'=1, seq_len: 'int'=16, layer_norm_eps: 'float'=1e-05, hidden_dropout_prob: 'float'=0.0, causal: 'bool'=False, mask_seq: 'int'=0) ->None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim=hidden_size, batch_size=batch_size, seq_len=seq_len, num_heads=num_attention_heads, qkv_bias=True, attn_drop=attention_dropout, proj_drop=hidden_dropout_prob, has_residual=False, causal=causal, mask_seq=mask_seq)

    def forward(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None, causal_attention_mask: 'Optional[Tensor]'=None, output_attentions: 'Optional[bool]'=False, residual: 'Optional[Tensor]'=None) ->Tensor:
        if residual is not None:
            return self.attn(hidden_states, residual)
        else:
            return self.attn(hidden_states)


class QuickGELUActivation(nn.Module):

    def forward(self, x: 'Tensor') ->Tensor:
        x1 = x * 1.702
        x1 = ops.sigmoid(x1)
        x = x * x1
        return x


class CLIPMLP(nn.Module):

    def __init__(self, in_features: 'int', hidden_features: 'Optional[int]'=None, out_features: 'Optional[int]'=None, act_layer: 'str'='GELU', drop: 'int'=0) ->None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, specialization='gelu')
        self.fc2 = nn.Linear(hidden_features, out_features, specialization='add')

    def forward(self, x: 'Tensor', residual: 'Tensor') ->Tensor:
        shape = x.shape()
        x = self.fc1(x)
        x = self.fc2(x, residual)
        return ops.reshape()(x, shape)


class CLIPMLPQuickGelu(nn.Module):

    def __init__(self, in_features: 'int', hidden_features: 'Optional[int]'=None, out_features: 'Optional[int]'=None) ->None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or out_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation_fn = QuickGELUActivation()
        self.fc2 = nn.Linear(hidden_features, out_features, specialization='add')

    def forward(self, x: 'Tensor', residual: 'Tensor') ->Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x, residual)
        return ops.reshape()(x, x.shape())


class CLIPEncoderLayer(nn.Module):
    ACT_LAYER_TO_CLIP_MLP_MAP = {'gelu': CLIPMLP, 'quick_gelu': CLIPMLPQuickGelu}

    def __init__(self, hidden_size: 'int'=768, num_attention_heads: 'int'=12, attention_dropout: 'float'=0.0, mlp_ratio: 'float'=4.0, batch_size: 'int'=1, seq_len: 'int'=16, causal: 'bool'=False, mask_seq: 'int'=0, act_layer: 'str'='gelu') ->None:
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = nn.CrossAttention(hidden_size, seq_len, seq_len, num_attention_heads, qkv_bias=True, causal=causal)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = self.ACT_LAYER_TO_CLIP_MLP_MAP[act_layer](hidden_size, int(hidden_size * mlp_ratio))
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'Tensor', output_attentions: 'Optional[bool]'=False) ->Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states, residual)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)
        return hidden_states


class CLIPEncoder(nn.Module):

    def __init__(self, num_hidden_layers: 'int'=12, output_attentions: 'bool'=False, output_hidden_states: 'bool'=False, use_return_dict: 'bool'=False, hidden_size: 'int'=768, num_attention_heads: 'int'=12, batch_size: 'int'=1, seq_len: 'int'=64, causal: 'bool'=False, mask_seq: 'int'=0, act_layer: 'str'='gelu') ->None:
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(hidden_size=hidden_size, num_attention_heads=num_attention_heads, batch_size=batch_size, seq_len=seq_len, causal=causal, mask_seq=mask_seq, act_layer=act_layer) for d in range(num_hidden_layers)])
        self.output_attentions = output_attentions,
        self.output_hidden_states = output_hidden_states,
        self.use_return_dict = use_return_dict

    def forward(self, inputs_embeds: 'Tensor', attention_mask: 'Optional[Tensor]'=None, causal_attention_mask: 'Optional[Tensor]'=None, output_attentions: 'Optional[bool]'=None, output_hidden_states: 'Optional[bool]'=None, return_dict: 'Optional[bool]'=None) ->Tensor:
        output_attentions = default(output_attentions, self.output_attentions)
        output_hidden_states = default(output_hidden_states, self.output_hidden_states)
        return_dict = default(return_dict, self.use_return_dict)
        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs
        return hidden_states


class CLIPTextEmbeddings(nn.Module):

    def __init__(self, hidden_size: 'int'=768, vocab_size: 'int'=49408, max_position_embeddings: 'int'=77, dtype: 'str'='float16') ->None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = hidden_size
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(shape=[vocab_size, hidden_size], dtype=dtype)
        self.position_embedding = nn.Embedding(shape=[max_position_embeddings, hidden_size], dtype=dtype)

    def forward(self, input_ids: 'Tensor', position_ids: 'Tensor', inputs_embeds: 'Optional[Tensor]'=None) ->Tensor:
        input_shape = ops.size()(input_ids)
        token_embedding = self.token_embedding.tensor()
        token_embedding = ops.reshape()(token_embedding, [1, self.vocab_size, self.embed_dim])
        token_embedding = ops.expand()(token_embedding, [input_shape[0], -1, -1])
        if inputs_embeds is None:
            inputs_embeds = ops.batch_gather()(token_embedding, input_ids)
        position_embedding = self.position_embedding.tensor()
        position_embedding = ops.reshape()(position_embedding, [1, self.max_position_embeddings, self.embed_dim])
        position_embedding = ops.expand()(position_embedding, [input_shape[0], -1, -1])
        position_embeddings = ops.batch_gather()(position_embedding, position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = ops.reshape()(embeddings, [input_shape[0], input_shape[1], -1])
        return embeddings


class CLIPTextTransformer(nn.Module):

    def __init__(self, hidden_size: 'int'=768, output_attentions: 'bool'=False, output_hidden_states: 'bool'=False, use_return_dict=False, num_hidden_layers: 'int'=12, num_attention_heads: 'int'=12, batch_size: 'int'=1, seq_len: 'int'=64, causal: 'bool'=False, mask_seq: 'int'=0, act_layer: 'str'='gelu') ->None:
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(hidden_size=hidden_size)
        self.encoder = CLIPEncoder(num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, num_attention_heads=num_attention_heads, batch_size=batch_size, seq_len=seq_len, causal=causal, mask_seq=mask_seq, act_layer=act_layer)
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

    def forward(self, input_ids: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, position_ids: 'Optional[Tensor]'=None, output_attentions: 'Optional[bool]'=None, output_hidden_states: 'Optional[bool]'=None, return_dict: 'Optional[bool]'=None) ->Tensor:
        output_attentions = default(output_attentions, self.output_attentions)
        output_hidden_states = default(output_hidden_states, self.output_hidden_states)
        return_dict = default(return_dict, self.use_return_dict)
        if input_ids is None:
            raise ValueError('input_ids must be specified!')
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        return last_hidden_state


class TimestepEmbedding(nn.Module):

    def __init__(self, channel: 'int', time_embed_dim: 'int', act_fn: 'str'='silu', dtype: 'str'='float16') ->None:
        super().__init__()
        self.linear_1 = nn.Linear(channel, time_embed_dim, specialization='swish', dtype=dtype)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype)

    def forward(self, sample: 'Tensor') ->Tensor:
        sample = self.linear_1(sample)
        sample = self.linear_2(sample)
        return sample


def get_timestep_embedding(timesteps: 'Tensor', embedding_dim: 'int', flip_sin_to_cos: 'bool'=False, downscale_freq_shift: 'float'=1, scale: 'float'=1, max_period: 'int'=10000, dtype: 'str'='float16', arange_name: 'str'='arange') ->Tensor:
    assert timesteps._rank() == 1, 'Timesteps should be a 1d-array'
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * Tensor(shape=[half_dim], dtype=dtype, name=arange_name)
    exponent = exponent * (1.0 / (half_dim - downscale_freq_shift))
    emb = ops.exp(exponent)
    emb = ops.reshape()(timesteps, [-1, 1]) * ops.reshape()(emb, [1, -1])
    emb = scale * emb
    if flip_sin_to_cos:
        emb = ops.concatenate()([ops.cos(emb), ops.sin(emb)], dim=-1)
    else:
        emb = ops.concatenate()([ops.sin(emb), ops.cos(emb)], dim=-1)
    return emb


class Timesteps(nn.Module):

    def __init__(self, num_channels: 'int', flip_sin_to_cos: 'bool', downscale_freq_shift: 'float', dtype: 'str'='float16', arange_name: 'str'='arange') ->None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.dtype = dtype
        self.arange_name = arange_name

    def forward(self, timesteps: 'Tensor') ->Tensor:
        t_emb = get_timestep_embedding(timesteps, self.num_channels, flip_sin_to_cos=self.flip_sin_to_cos, downscale_freq_shift=self.downscale_freq_shift, dtype=self.dtype, arange_name=self.arange_name)
        return t_emb


class Upsample2D(nn.Module):

    def __init__(self, channels: 'int', use_conv: 'bool'=False, use_conv_transpose: 'bool'=False, out_channels: 'Optional[int]'=None, name: 'str'='conv', dtype: 'str'='float16') ->None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2dBias(channels, self.out_channels, 4, 2, 1, dtype=dtype)
        elif use_conv:
            conv = nn.Conv2dBias(self.channels, self.out_channels, 3, 1, 1, dtype=dtype)
        if name == 'conv':
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, x: 'Tensor') ->Tensor:
        if self.use_conv_transpose:
            return self.conv(x)
        x = nn.Upsampling2d(scale_factor=2.0, mode='nearest')(x)
        if self.use_conv:
            if self.name == 'conv':
                x = self.conv(x)
            else:
                x = self.Conv2d_0(x)
        return x


def get_shape(x):
    shape = [it.value() for it in x._attrs['shape']]
    return shape


class Downsample2D(nn.Module):

    def __init__(self, channels: 'int', use_conv: 'bool'=False, out_channels: 'Optional[int]'=None, padding: 'int'=1, name: 'str'='conv', dtype: 'str'='float16') ->None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.dtype = dtype
        if use_conv:
            conv = nn.Conv2dBias(self.channels, self.out_channels, 3, stride=stride, dtype=dtype, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)
        if name == 'conv':
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == 'Conv2d_0':
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: 'Tensor') ->Tensor:
        if self.use_conv and self.padding == 0:
            shape = get_shape(hidden_states)
            padding = ops.full()([0, 1, 0, 0], 0.0, dtype=self.dtype)
            padding._attrs['shape'][0] = shape[0]
            padding._attrs['shape'][2] = shape[2]
            padding._attrs['shape'][3] = shape[3]
            hidden_states = ops.concatenate()([hidden_states, padding], dim=1)
            shape = get_shape(hidden_states)
            padding = ops.full()([0, 0, 1, 0], 0.0, dtype=self.dtype)
            padding._attrs['shape'][0] = shape[0]
            padding._attrs['shape'][1] = shape[1]
            padding._attrs['shape'][3] = shape[3]
            hidden_states = ops.concatenate()([hidden_states, padding], dim=2)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlock2D(nn.Module):

    def __init__(self, *, in_channels: int, out_channels: Optional[int]=None, conv_shortcut: bool=False, dropout: float=0.0, temb_channels: Optional[int]=512, groups: int=32, groups_out: Optional[int]=None, pre_norm: bool=True, eps: float=1e-06, non_linearity: str='swish', time_embedding_norm: str='default', kernel: Optional[int]=None, output_scale_factor: float=1.0, use_nin_shortcut: Optional[bool]=None, up: bool=False, down: bool=False, dtype: str='float16') ->None:
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        if groups_out is None:
            groups_out = groups
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True, use_swish=True, dtype=dtype)
        self.conv1 = nn.Conv2dBias(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels, dtype=dtype)
        else:
            self.time_emb_proj = None
        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True, use_swish=True, dtype=dtype)
        self.dropout = nn.Dropout(dropout, dtype=dtype)
        self.conv2 = nn.Conv2dBias(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.upsample = self.downsample = None
        self.use_nin_shortcut = self.in_channels != self.out_channels if use_nin_shortcut is None else use_nin_shortcut
        if self.use_nin_shortcut:
            self.conv_shortcut = nn.Conv2dBias(in_channels, out_channels, 1, 1, 0, dtype=dtype)
        else:
            self.conv_shortcut = None

    def forward(self, x: 'Tensor', temb: 'Optional[Tensor]'=None) ->Tensor:
        hidden_states = x
        hidden_states = self.norm1(hidden_states)
        if self.upsample is not None:
            x = self.upsample(x)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            x = self.downsample(x)
            hidden_states = self.downsample(hidden_states)
        hidden_states = self.conv1(hidden_states)
        bs, _, _, dim = hidden_states.shape()
        if temb is not None:
            temb = self.time_emb_proj(ops.silu(temb))
            bs, dim = temb.shape()
            temb = ops.reshape()(temb, [bs, 1, 1, dim])
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        out = hidden_states + x
        return out


class UNetMidBlock2DCrossAttn(nn.Module):

    def __init__(self, in_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, transformer_layers_per_block=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels: 'int'=1, attention_type: 'str'='default', output_scale_factor: 'float'=1.0, cross_attention_dim: 'int'=1280, use_linear_projection: 'bool'=False, dtype: 'str'='float16') ->None:
        super().__init__()
        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(SpatialTransformer(in_channels, attn_num_head_channels, in_channels // attn_num_head_channels, depth=transformer_layers_per_block, context_dim=cross_attention_dim, use_linear_projection=use_linear_projection, dtype=dtype))
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: 'Tensor', temb: 'Optional[Tensor]'=None, encoder_hidden_states: 'Optional[Tensor]'=None) ->Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class CrossAttnDownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, transformer_layers_per_block: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels=1, cross_attention_dim=1280, attention_type: 'str'='default', output_scale_factor: 'float'=1.0, downsample_padding: 'int'=1, add_downsample: 'bool'=True, use_linear_projection: 'bool'=False, only_cross_attention: 'bool'=False, dtype: 'str'='float16') ->None:
        super().__init__()
        resnets = []
        attentions = []
        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype))
            attentions.append(SpatialTransformer(out_channels, attn_num_head_channels, out_channels // attn_num_head_channels, depth=transformer_layers_per_block, context_dim=cross_attention_dim, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, dtype=dtype))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op', dtype=dtype)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: 'Tensor', temb: 'Optional[Tensor]'=None, encoder_hidden_states: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tuple[Tensor]]:
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=encoder_hidden_states)
            output_states += hidden_states,
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += hidden_states,
        return hidden_states, output_states


class DownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor: 'float'=1.0, add_downsample: 'bool'=True, downsample_padding: 'int'=1, dtype: 'str'='float16') ->None:
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op', dtype=dtype)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: 'Tensor', temb: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tuple[Tensor]]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += hidden_states,
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += hidden_states,
        return hidden_states, output_states


class DownEncoderBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor: 'float'=1.0, add_downsample: 'bool'=True, downsample_padding: 'int'=1, dtype: 'str'='float16') ->None:
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=None, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op', dtype=dtype)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: 'Tensor') ->Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


def get_down_block(down_block_type: 'str', num_layers: 'int', in_channels: 'int', out_channels: 'int', temb_channels: 'int', add_downsample: 'bool', resnet_eps: 'float', resnet_act_fn: 'str', attn_num_head_channels: 'int', transformer_layers_per_block: 'int'=1, cross_attention_dim: 'Optional[int]'=None, downsample_padding: 'Optional[int]'=None, use_linear_projection: 'Optional[bool]'=False, only_cross_attention: 'Optional[bool]'=False, resnet_groups: 'int'=32, dtype: 'str'='float16') ->Any:
    down_block_type = down_block_type[7:] if down_block_type.startswith('UNetRes') else down_block_type
    if down_block_type == 'DownBlock2D':
        return DownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, dtype=dtype)
    elif down_block_type == 'CrossAttnDownBlock2D':
        if cross_attention_dim is None:
            raise ValueError('cross_attention_dim must be specified for CrossAttnDownBlock2D')
        return CrossAttnDownBlock2D(num_layers=num_layers, transformer_layers_per_block=transformer_layers_per_block, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, cross_attention_dim=cross_attention_dim, attn_num_head_channels=attn_num_head_channels, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, dtype=dtype)
    elif down_block_type == 'DownEncoderBlock2D':
        return DownEncoderBlock2D(in_channels=in_channels, out_channels=out_channels, num_layers=num_layers, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, output_scale_factor=1.0, add_downsample=add_downsample, downsample_padding=downsample_padding, dtype=dtype)


class CrossAttnUpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', prev_output_channel: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, transformer_layers_per_block: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels: 'int'=1, cross_attention_dim: 'int'=1280, attention_type: 'str'='default', output_scale_factor: 'float'=1.0, downsample_padding: 'int'=1, add_upsample: 'bool'=True, use_linear_projection: 'bool'=False, only_cross_attention: 'bool'=False, dtype: 'str'='float16') ->None:
        super().__init__()
        resnets = []
        attentions = []
        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype))
            attentions.append(SpatialTransformer(out_channels, attn_num_head_channels, out_channels // attn_num_head_channels, depth=transformer_layers_per_block, context_dim=cross_attention_dim, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, dtype=dtype))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels, dtype=dtype)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: 'Tensor', res_hidden_states_tuple: 'Tuple[Tensor]', temb: 'Optional[Tensor]'=None, encoder_hidden_states: 'Optional[Tensor]'=None) ->Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = ops.concatenate()([hidden_states, res_hidden_states], dim=-1)
            hidden_states = resnet(hidden_states, temb=temb)
            hidden_states = attn(hidden_states, context=encoder_hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class UpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', prev_output_channel: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor: 'float'=1.0, add_upsample: 'bool'=True, dtype: 'str'='float16') ->None:
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels, dtype=dtype)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: 'Tensor', res_hidden_states_tuple: 'Tuple[Tensor]', temb: 'Optional[Tensor]'=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = ops.concatenate()([hidden_states, res_hidden_states], dim=-1)
            hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor: 'float'=1.0, add_upsample: 'bool'=True, dtype: 'str'='float16') ->None:
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=input_channels, out_channels=out_channels, temb_channels=None, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels, dtype=dtype)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: 'Tensor') ->Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


def get_up_block(up_block_type: 'str', num_layers: 'int', in_channels: 'int', out_channels: 'int', prev_output_channel: 'Optional[int]', temb_channels: 'Optional[int]', add_upsample: 'bool', resnet_eps: 'float', resnet_act_fn: 'str', attn_num_head_channels: 'Optional[int]', transformer_layers_per_block: 'int'=1, cross_attention_dim: 'Optional[int]'=None, use_linear_projection: 'Optional[bool]'=False, only_cross_attention: 'bool'=False, dtype: 'str'='float16') ->Any:
    up_block_type = up_block_type[7:] if up_block_type.startswith('UNetRes') else up_block_type
    if up_block_type == 'UpBlock2D':
        return UpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, dtype=dtype)
    elif up_block_type == 'CrossAttnUpBlock2D':
        if cross_attention_dim is None:
            raise ValueError('cross_attention_dim must be specified for CrossAttnUpBlock2D')
        return CrossAttnUpBlock2D(transformer_layers_per_block=transformer_layers_per_block, num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, cross_attention_dim=cross_attention_dim, attn_num_head_channels=attn_num_head_channels, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, dtype=dtype)
    elif up_block_type == 'UpDecoderBlock2D':
        return UpDecoderBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, dtype=dtype)
    raise ValueError(f'{up_block_type} does not exist.')


class UNet2DConditionModel(nn.Module):

    def __init__(self, sample_size: 'Optional[int]'=None, in_channels: 'int'=4, out_channels: 'int'=4, center_input_sample: 'bool'=False, flip_sin_to_cos: 'bool'=True, freq_shift: 'int'=0, down_block_types: 'Tuple[str, str, str, str]'=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'), up_block_types: 'Tuple[str, str, str, str]'=('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'), block_out_channels: 'Tuple[int, int, int, int]'=(320, 640, 1280, 1280), layers_per_block: 'int'=2, downsample_padding: 'int'=1, mid_block_scale_factor: 'float'=1, act_fn: 'str'='silu', norm_num_groups: 'int'=32, norm_eps: 'float'=1e-05, cross_attention_dim: 'int'=1280, attention_head_dim: 'Union[int, Tuple[int]]'=8, use_linear_projection: 'bool'=False, class_embed_type: 'Optional[str]'=None, num_class_embeds: 'Optional[int]'=None, only_cross_attention: 'List[bool]'=[True, True, True, False], conv_in_kernel: 'int'=3, dtype: 'str'='float16', time_embedding_dim: 'Optional[int]'=None, projection_class_embeddings_input_dim: 'Optional[int]'=None, addition_embed_type: 'Optional[str]'=None, addition_time_embed_dim: 'Optional[int]'=None, transformer_layers_per_block: 'int'=1) ->None:
        super().__init__()
        self.center_input_sample = center_input_sample
        self.sample_size = sample_size
        self.time_embedding_dim = time_embedding_dim
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        self.in_channels = in_channels
        if in_channels >= 1 and in_channels <= 4:
            in_channels = 4
        elif in_channels > 4 and in_channels <= 8:
            in_channels = 8
        elif in_channels > 8 and in_channels <= 12:
            in_channels = 12
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2dBias(in_channels, block_out_channels[0], 3, 1, conv_in_padding, dtype=dtype)
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift, dtype=dtype, arange_name='arange')
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, dtype=dtype)
        self.class_embed_type = class_embed_type
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding([num_class_embeds, time_embed_dim], dtype=dtype)
        elif class_embed_type == 'timestep':
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, dtype=dtype)
        elif class_embed_type == 'identity':
            self.class_embedding = nn.Identity(dtype=dtype)
        else:
            self.class_embedding = None
        if addition_embed_type == 'text_time':
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim, dtype=dtype)
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(down_block_type, num_layers=layers_per_block, transformer_layers_per_block=transformer_layers_per_block[i], in_channels=input_channel, out_channels=output_channel, temb_channels=time_embed_dim, add_downsample=not is_final_block, resnet_eps=norm_eps, resnet_act_fn=act_fn, attn_num_head_channels=attention_head_dim[i], cross_attention_dim=cross_attention_dim, downsample_padding=downsample_padding, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention[i], dtype=dtype)
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock2DCrossAttn(transformer_layers_per_block=transformer_layers_per_block[-1], in_channels=block_out_channels[-1], temb_channels=time_embed_dim, resnet_eps=norm_eps, resnet_act_fn=act_fn, output_scale_factor=mid_block_scale_factor, resnet_time_scale_shift='default', cross_attention_dim=cross_attention_dim, attn_num_head_channels=attention_head_dim[-1], resnet_groups=norm_num_groups, use_linear_projection=use_linear_projection, dtype=dtype)
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(up_block_type, num_layers=layers_per_block + 1, transformer_layers_per_block=reversed_transformer_layers_per_block[i], in_channels=input_channel, out_channels=output_channel, prev_output_channel=prev_output_channel, temb_channels=time_embed_dim, add_upsample=not is_final_block, resnet_eps=norm_eps, resnet_act_fn=act_fn, attn_num_head_channels=reversed_attention_head_dim[i], cross_attention_dim=cross_attention_dim, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention[i], dtype=dtype)
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps, use_swish=True, dtype=dtype)
        self.conv_out = nn.Conv2dBias(block_out_channels[0], out_channels, 3, 1, 1, dtype=dtype)

    def forward(self, sample: 'Tensor', timesteps: 'Tensor', encoder_hidden_states: 'Tensor', down_block_residual_0: 'Optional[Tensor]'=None, down_block_residual_1: 'Optional[Tensor]'=None, down_block_residual_2: 'Optional[Tensor]'=None, down_block_residual_3: 'Optional[Tensor]'=None, down_block_residual_4: 'Optional[Tensor]'=None, down_block_residual_5: 'Optional[Tensor]'=None, down_block_residual_6: 'Optional[Tensor]'=None, down_block_residual_7: 'Optional[Tensor]'=None, down_block_residual_8: 'Optional[Tensor]'=None, down_block_residual_9: 'Optional[Tensor]'=None, down_block_residual_10: 'Optional[Tensor]'=None, down_block_residual_11: 'Optional[Tensor]'=None, mid_block_residual: 'Optional[Tensor]'=None, class_labels: 'Optional[Tensor]'=None, add_embeds: 'Optional[Tensor]'=None, return_dict: 'bool'=True) ->Tensor:
        down_block_additional_residuals = down_block_residual_0, down_block_residual_1, down_block_residual_2, down_block_residual_3, down_block_residual_4, down_block_residual_5, down_block_residual_6, down_block_residual_7, down_block_residual_8, down_block_residual_9, down_block_residual_10, down_block_residual_11
        mid_block_additional_residual = mid_block_residual
        if down_block_additional_residuals[0] is None:
            down_block_additional_residuals = None
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError('class_labels should be provided when num_class_embeds > 0')
            if self.class_embed_type == 'timestep':
                class_labels = self.time_proj(class_labels)
            class_emb = ops.batch_gather()(self.class_embedding.weight.tensor(), class_labels)
            emb = emb + class_emb
        if add_embeds is not None:
            aug_emb = self.add_embedding(add_embeds)
            emb = emb + aug_emb
        if self.in_channels < 4:
            sample = ops.pad_last_dim(4, 4)(sample)
        elif self.in_channels > 4 and self.in_channels < 8:
            sample = ops.pad_last_dim(4, 8)(sample)
        elif self.in_channels > 8 and self.in_channels < 12:
            sample = ops.pad_last_dim(4, 12)(sample)
        sample = self.conv_in(sample)
        down_block_res_samples = sample,
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, 'attentions') and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(down_block_res_samples, down_block_additional_residuals):
                down_block_additional_residual._attrs['shape'] = down_block_res_sample._attrs['shape']
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples += down_block_res_sample,
            down_block_res_samples = new_down_block_res_samples
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        if mid_block_additional_residual is not None:
            mid_block_additional_residual._attrs['shape'] = sample._attrs['shape']
            sample += mid_block_additional_residual
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if hasattr(upsample_block, 'attentions') and upsample_block.attentions is not None:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, encoder_hidden_states=encoder_hidden_states)
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)
        sample = self.conv_norm_out(sample)
        sample = self.conv_out(sample)
        return sample


class UNetMidBlock2D(nn.Module):

    def __init__(self, batch_size: 'int', height: 'int', width: 'int', in_channels: 'int', temb_channels: 'Optional[int]', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels: 'Optional[int]'=1, attention_type: 'str'='default', output_scale_factor: 'float'=1.0, dtype: 'str'='float16') ->None:
        super().__init__()
        if attention_type != 'default':
            raise NotImplementedError(f'attention_type must be default! current value: {attention_type}')
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock(batch_size, height, width, in_channels, num_head_channels=attn_num_head_channels, rescale_output_factor=output_scale_factor, eps=resnet_eps, num_groups=resnet_groups, dtype=dtype))
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, dtype=dtype))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: 'Tensor', temb: 'Optional[Tensor]'=None, encoder_states: 'Optional[Tensor]'=None) ->Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class Decoder(nn.Module):

    def __init__(self, batch_size: 'int', height: 'int', width: 'int', in_channels: 'int'=3, out_channels: 'int'=3, up_block_types: 'Tuple[str]'=('UpDecoderBlock2D',), block_out_channels: 'Tuple[int]'=(64,), layers_per_block: 'int'=2, act_fn: 'str'='silu', dtype: 'str'='float16') ->None:
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2dBias(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.mid_block = UNetMidBlock2D(batch_size, height, width, in_channels=block_out_channels[-1], resnet_eps=1e-06, resnet_act_fn=act_fn, output_scale_factor=1, resnet_time_scale_shift='default', attn_num_head_channels=None, resnet_groups=32, temb_channels=None, dtype=dtype)
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(up_block_type, num_layers=self.layers_per_block + 1, in_channels=prev_output_channel, out_channels=output_channel, prev_output_channel=None, temb_channels=None, add_upsample=not is_final_block, resnet_eps=1e-06, resnet_act_fn=act_fn, attn_num_head_channels=None, dtype=dtype)
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        num_groups_out = 32
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=1e-06, use_swish=True, dtype=dtype)
        self.conv_out = nn.Conv2dBias(block_out_channels[0], out_channels, kernel_size=3, padding=1, stride=1, dtype=dtype)

    def forward(self, z: 'Tensor') ->Tensor:
        sample = z
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_out(sample)
        return sample


class Encoder(nn.Module):

    def __init__(self, batch_size: 'int', height: 'int', width: 'int', in_channels: 'int'=3, out_channels: 'int'=3, down_block_types: 'Tuple[str]'=('DownEncoderBlock2D',), block_out_channels: 'Tuple[int]'=(64,), layers_per_block: 'int'=2, norm_num_groups: 'int'=32, act_fn: 'str'='silu', double_z: 'bool'=True, dtype: 'str'='float16') ->None:
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2dBiasFewChannels(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(down_block_type, num_layers=self.layers_per_block, in_channels=input_channel, out_channels=output_channel, add_downsample=not is_final_block, resnet_eps=1e-06, downsample_padding=0, resnet_act_fn=act_fn, resnet_groups=norm_num_groups, attn_num_head_channels=None, temb_channels=None, dtype=dtype)
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock2D(batch_size, height, width, in_channels=block_out_channels[-1], resnet_eps=1e-06, resnet_act_fn=act_fn, output_scale_factor=1, resnet_time_scale_shift='default', attn_num_head_channels=None, resnet_groups=norm_num_groups, temb_channels=None, dtype=dtype)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-06, dtype=dtype)
        self.conv_act = ops.silu
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2dBias(block_out_channels[-1], conv_out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)

    def forward(self, x: 'Tensor') ->Tensor:
        sample = x
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class AutoencoderKL(nn.Module):

    def __init__(self, batch_size: 'int', height: 'int', width: 'int', in_channels: 'int'=3, out_channels: 'int'=3, down_block_types: 'Tuple[str]'=('DownEncoderBlock2D',), up_block_types: 'Tuple[str]'=('UpDecoderBlock2D',), block_out_channels: 'Tuple[int]'=(64,), layers_per_block: 'int'=1, act_fn: 'str'='silu', latent_channels: 'int'=4, norm_num_groups: 'int'=32, sample_size: 'int'=32, dtype: 'str'='float16') ->None:
        super().__init__()
        self.decoder = Decoder(batch_size, height, width, in_channels=latent_channels, out_channels=out_channels, up_block_types=up_block_types, block_out_channels=block_out_channels, layers_per_block=layers_per_block, act_fn=act_fn, dtype=dtype)
        self.post_quant_conv = nn.Conv2dBias(latent_channels, latent_channels, kernel_size=1, stride=1, padding=0, dtype=dtype)
        self.encoder = Encoder(batch_size, height, width, in_channels=in_channels, out_channels=latent_channels, down_block_types=down_block_types, block_out_channels=block_out_channels, layers_per_block=layers_per_block, act_fn=act_fn, norm_num_groups=norm_num_groups, double_z=True, dtype=dtype)
        self.quant_conv = nn.Conv2dBias(2 * latent_channels, 2 * latent_channels, kernel_size=1, stride=1, padding=0, dtype=dtype)

    def forward(self):
        pass

    def decode(self, z: 'Tensor', return_dict: 'bool'=True) ->Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def encode(self, x: 'Tensor', sample: 'Tensor'=None, return_dict: 'bool'=True, deterministic: 'bool'=False) ->Tensor:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if sample is None:
            return moments
        mean, logvar = ops.chunk()(moments, 2, dim=3)
        logvar = ops.clamp()(logvar, -30.0, 20.0)
        std = ops.exp(0.5 * logvar)
        if deterministic:
            std = Tensor(mean.shape(), value=0.0, dtype=mean._attrs['dtype'])
        sample._attrs['shape'] = mean._attrs['shape']
        std._attrs['shape'] = mean._attrs['shape']
        z = mean + std * sample
        return z


class RRDBNet(nn.Module):

    def __init__(self, state_dict, norm=None, act: 'str'='leakyrelu', upsampler: 'str'='upconv', mode: 'str'='CNA') ->None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.
        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.
        This network supports model files from both new and old-arch.
        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super(RRDBNet, self).__init__()
        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode
        self.state_map = {'model.0.weight': ('conv_first.weight',), 'model.0.bias': ('conv_first.bias',), 'model.1.sub./NB/.weight': ('trunk_conv.weight', 'conv_body.weight'), 'model.1.sub./NB/.bias': ('trunk_conv.bias', 'conv_body.bias'), 'model.3.weight': ('upconv1.weight', 'conv_up1.weight'), 'model.3.bias': ('upconv1.bias', 'conv_up1.bias'), 'model.6.weight': ('upconv2.weight', 'conv_up2.weight'), 'model.6.bias': ('upconv2.bias', 'conv_up2.bias'), 'model.8.weight': ('HRconv.weight', 'conv_hr.weight'), 'model.8.bias': ('HRconv.bias', 'conv_hr.bias'), 'model.10.weight': ('conv_last.weight',), 'model.10.bias': ('conv_last.bias',), 'model.1.sub.\\1.RDB\\2.conv\\3.0.\\4': ('RRDB_trunk\\.(\\d+)\\.RDB(\\d)\\.conv(\\d+)\\.(weight|bias)', 'body\\.(\\d+)\\.rdb(\\d)\\.conv(\\d+)\\.(weight|bias)')}
        if 'params_ema' in self.state:
            self.state = self.state['params_ema']
        self.num_blocks = self.get_num_blocks()
        self.plus = any('conv1x1' in k for k in self.state.keys())
        self.state = self.new_to_old_arch(self.state)
        self.key_arr = list(self.state.keys())
        self.in_nc = self.state[self.key_arr[0]].shape[1]
        self.out_nc = self.state[self.key_arr[-1]].shape[0]
        self.scale = self.get_scale()
        self.num_filters = self.state[self.key_arr[0]].shape[0]
        c2x2 = False
        if self.state['model.0.weight'].shape[-2] == 2:
            c2x2 = True
            self.scale = math.ceil(self.scale ** (1.0 / 3))
        if self.in_nc in (self.out_nc * 4, self.out_nc * 16) and self.out_nc in (self.in_nc / 4, self.in_nc / 16):
            self.shuffle_factor = int(math.sqrt(self.in_nc / self.out_nc))
        else:
            self.shuffle_factor = None
        upsample_block = {'upconv': B.upconv_block, 'pixel_shuffle': B.pixelshuffle_block}.get(self.upsampler)
        if upsample_block is None:
            raise NotImplementedError(f'Upsample mode [{self.upsampler}] is not found')
        if self.scale == 3:
            upsample_blocks = upsample_block(in_nc=self.num_filters, out_nc=self.num_filters, upscale_factor=3, act_type=self.act, c2x2=c2x2)
        else:
            upsample_blocks = [upsample_block(in_nc=self.num_filters, out_nc=self.num_filters, act_type=self.act, c2x2=c2x2) for _ in range(int(math.log(self.scale, 2)))]
        self.model = B.sequential(B.conv_block(in_nc=self.in_nc, out_nc=self.num_filters, kernel_size=3, norm_type=None, act_type=None, c2x2=c2x2), B.ShortcutBlock(B.sequential(*[B.RRDB(nf=self.num_filters, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=self.norm, act_type=self.act, mode='CNA', plus=self.plus, c2x2=c2x2) for _ in range(self.num_blocks)], B.conv_block(in_nc=self.num_filters, out_nc=self.num_filters, kernel_size=3, norm_type=self.norm, act_type=None, mode=self.mode, c2x2=c2x2))), *upsample_blocks, B.conv_block(in_nc=self.num_filters, out_nc=self.num_filters, kernel_size=3, norm_type=None, act_type=self.act, c2x2=c2x2), B.conv_block(in_nc=self.num_filters, out_nc=self.out_nc, kernel_size=3, norm_type=None, act_type=None, c2x2=c2x2))
        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state):
        """Convert a new-arch model state dictionary to an old-arch dictionary."""
        if 'params_ema' in state:
            state = state['params_ema']
        if 'conv_first.weight' not in state:
            return state
        for kind in ('weight', 'bias'):
            self.state_map[f'model.1.sub.{self.num_blocks}.{kind}'] = self.state_map[f'model.1.sub./NB/.{kind}']
            del self.state_map[f'model.1.sub./NB/.{kind}']
        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if '\\1' in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                elif new_key in state:
                    old_state[old_key] = state[new_key]

        def compare(item1, item2):
            parts1 = item1.split('.')
            parts2 = item2.split('.')
            int1 = int(parts1[1])
            int2 = int(parts2[1])
            return int1 - int2
        sorted_keys = sorted(old_state.keys(), key=functools.cmp_to_key(compare))
        out_dict = OrderedDict((k, old_state[k]) for k in sorted_keys)
        return out_dict

    def get_scale(self, min_part: 'int'=6) ->int:
        n = 0
        for part in list(self.state):
            parts = part.split('.')[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == 'weight':
                    n += 1
        return 2 ** n

    def get_num_blocks(self) ->int:
        nbs = []
        state_keys = self.state_map['model.1.sub.\\1.RDB\\2.conv\\3.0.\\4'] + ('model\\.\\d+\\.sub\\.(\\d+)\\.RDB(\\d+)\\.conv(\\d+)\\.0\\.(weight|bias)',)
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x):
        if self.shuffle_factor:
            x = torch.pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
        return self.model(x)


class Get_gradient_nopadding(nn.Module):

    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-06)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x


class SPSRNet(nn.Module):

    def __init__(self, state_dict, norm=None, act: 'str'='leakyrelu', upsampler: 'str'='upconv', mode: 'str'='CNA'):
        super(SPSRNet, self).__init__()
        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode
        self.num_blocks = self.get_num_blocks()
        self.in_nc = self.state['model.0.weight'].shape[1]
        self.out_nc = self.state['f_HR_conv1.0.bias'].shape[0]
        self.scale = self.get_scale(4)
        None
        self.num_filters = self.state['model.0.weight'].shape[0]
        n_upscale = int(math.log(self.scale, 2))
        if self.scale == 3:
            n_upscale = 1
        fea_conv = B.conv_block(self.in_nc, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(self.num_filters, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm, act_type=act, mode='CNA') for _ in range(self.num_blocks)]
        LR_conv = B.conv_block(self.num_filters, self.num_filters, kernel_size=3, norm_type=norm, act_type=None, mode=mode)
        if upsampler == 'upconv':
            upsample_block = B.upconv_block
        elif upsampler == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(f'upsample mode [{upsampler}] is not found')
        if self.scale == 3:
            a_upsampler = upsample_block(self.num_filters, self.num_filters, 3, act_type=act)
        else:
            a_upsampler = [upsample_block(self.num_filters, self.num_filters, act_type=act) for _ in range(n_upscale)]
        self.HR_conv0_new = B.conv_block(self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=act)
        self.HR_conv1_new = B.conv_block(self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        self.model = B.sequential(fea_conv, B.ShortcutBlockSPSR(B.sequential(*rb_blocks, LR_conv)), *a_upsampler, self.HR_conv0_new)
        self.get_g_nopadding = Get_gradient_nopadding()
        self.b_fea_conv = B.conv_block(self.in_nc, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        self.b_concat_1 = B.conv_block(2 * self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_1 = B.RRDB(self.num_filters * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm, act_type=act, mode='CNA')
        self.b_concat_2 = B.conv_block(2 * self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_2 = B.RRDB(self.num_filters * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm, act_type=act, mode='CNA')
        self.b_concat_3 = B.conv_block(2 * self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_3 = B.RRDB(self.num_filters * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm, act_type=act, mode='CNA')
        self.b_concat_4 = B.conv_block(2 * self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_4 = B.RRDB(self.num_filters * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm, act_type=act, mode='CNA')
        self.b_LR_conv = B.conv_block(self.num_filters, self.num_filters, kernel_size=3, norm_type=norm, act_type=None, mode=mode)
        if upsampler == 'upconv':
            upsample_block = B.upconv_block
        elif upsampler == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(f'upsample mode [{upsampler}] is not found')
        if self.scale == 3:
            b_upsampler = upsample_block(self.num_filters, self.num_filters, 3, act_type=act)
        else:
            b_upsampler = [upsample_block(self.num_filters, self.num_filters, act_type=act) for _ in range(n_upscale)]
        b_HR_conv0 = B.conv_block(self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=act)
        b_HR_conv1 = B.conv_block(self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        self.b_module = B.sequential(*b_upsampler, b_HR_conv0, b_HR_conv1)
        self.conv_w = B.conv_block(self.num_filters, self.out_nc, kernel_size=1, norm_type=None, act_type=None)
        self.f_concat = B.conv_block(self.num_filters * 2, self.num_filters, kernel_size=3, norm_type=None, act_type=None)
        self.f_block = B.RRDB(self.num_filters * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm, act_type=act, mode='CNA')
        self.f_HR_conv0 = B.conv_block(self.num_filters, self.num_filters, kernel_size=3, norm_type=None, act_type=act)
        self.f_HR_conv1 = B.conv_block(self.num_filters, self.out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.load_state_dict(self.state, strict=False)

    def get_scale(self, min_part: 'int'=4) ->int:
        n = 0
        for part in list(self.state):
            parts = part.split('.')
            if len(parts) == 3:
                part_num = int(parts[1])
                if part_num > min_part and parts[0] == 'model' and parts[2] == 'weight':
                    n += 1
        return 2 ** n

    def get_num_blocks(self) ->int:
        nb = 0
        for part in list(self.state):
            parts = part.split('.')
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == 'sub':
                nb = int(parts[3])
        return nb

    def forward(self, x):
        x_grad = self.get_g_nopadding(x)
        x = self.model[0](x)
        x, block_list = self.model[1](x)
        x_ori = x
        for i in range(5):
            x = block_list[i](x)
        x_fea1 = x
        for i in range(5):
            x = block_list[i + 5](x)
        x_fea2 = x
        for i in range(5):
            x = block_list[i + 10](x)
        x_fea3 = x
        for i in range(5):
            x = block_list[i + 15](x)
        x_fea4 = x
        x = block_list[20:](x)
        x = x_ori + x
        x = self.model[2:](x)
        x = self.HR_conv1_new(x)
        x_b_fea = self.b_fea_conv(x_grad)
        x_cat_1 = torch.cat([x_b_fea, x_fea1], dim=1)
        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)
        x_cat_2 = torch.cat([x_cat_1, x_fea2], dim=1)
        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)
        x_cat_3 = torch.cat([x_cat_2, x_fea3], dim=1)
        x_cat_3 = self.b_block_3(x_cat_3)
        x_cat_3 = self.b_concat_3(x_cat_3)
        x_cat_4 = torch.cat([x_cat_3, x_fea4], dim=1)
        x_cat_4 = self.b_block_4(x_cat_4)
        x_cat_4 = self.b_concat_4(x_cat_4)
        x_cat_4 = self.b_LR_conv(x_cat_4)
        x_cat_4 = x_cat_4 + x_b_fea
        x_branch = self.b_module(x_cat_4)
        x_branch_d = x_branch
        x_f_cat = torch.cat([x_branch_d, x], dim=1)
        x_f_cat = self.f_block(x_f_cat)
        x_out = self.f_concat(x_f_cat)
        x_out = self.f_HR_conv0(x_out)
        x_out = self.f_HR_conv1(x_out)
        return x_out


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, state_dict, act_type: 'str'='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.act_type = act_type
        self.state = state_dict
        if 'params' in self.state:
            self.state = self.state['params']
        self.key_arr = list(self.state.keys())
        self.num_in_ch = self.get_in_nc()
        self.num_feat = self.get_num_feats()
        self.num_conv = self.get_num_conv()
        self.num_out_ch = self.num_in_ch
        self.scale = self.get_scale()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(self.num_in_ch, self.num_feat, 3, 1, 1))
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=self.num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)
        for _ in range(self.num_conv):
            self.body.append(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=self.num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)
        self.body.append(nn.Conv2d(self.num_feat, self.pixelshuffle_shape, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(self.scale)
        self.load_state_dict(self.state, strict=False)

    def get_num_conv(self) ->int:
        return (int(self.key_arr[-1].split('.')[1]) - 2) // 2

    def get_num_feats(self) ->int:
        return self.state[self.key_arr[0]].shape[0]

    def get_in_nc(self) ->int:
        return self.state[self.key_arr[0]].shape[1]

    def get_scale(self) ->int:
        self.pixelshuffle_shape = self.state[self.key_arr[-1]].shape[0]
        self.num_out_ch = self.num_in_ch
        scale = math.sqrt(self.pixelshuffle_shape / self.num_out_ch)
        if scale - int(scale) > 0:
            None
        scale = int(scale)
        return scale

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        out += base
        return out


class ConcatBlock(nn.Module):

    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):

    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlockSPSR(nn.Module):

    def __init__(self, submodule):
        super(ShortcutBlockSPSR, self).__init__()
        self.sub = submodule

    def forward(self, x):
        return x, self.sub

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block_2c2(in_nc, out_nc, act_type='relu'):
    return sequential(nn.Conv2d(in_nc, out_nc, kernel_size=2, padding=1), nn.Conv2d(out_nc, out_nc, kernel_size=2, padding=0), act(act_type) if act_type else None)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', c2x2=False):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    if c2x2:
        return conv_block_2c2(in_nc, out_nc, act_type=act_type)
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
            {Rakotonirina} and A. {Rasoanaivo}

    Args:
        nf (int): Channel number of intermediate features (num_feat).
        gc (int): Channels for each growth (num_grow_ch: growth channel,
            i.e. intermediate channels).
        convtype (str): the type of convolution to use. Default: 'Conv2D'
        gaussian_noise (bool): enable the ESRGAN+ gaussian noise (no new
            trainable parameters)
        plus (bool): enable the additional residual paths from ESRGAN+
            (adds trainable parameters)
    """

    def __init__(self, nf=64, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA', plus=False, c2x2=False):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1x1 = conv1x1(nf, gc) if plus else None
        self.conv1 = conv_block(nf, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode, c2x2=c2x2)
        self.conv2 = conv_block(nf + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode, c2x2=c2x2)
        self.conv3 = conv_block(nf + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode, c2x2=c2x2)
        self.conv4 = conv_block(nf + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode, c2x2=c2x2)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nf + 4 * gc, nf, 3, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=last_act, mode=mode, c2x2=c2x2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(self, nf, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero', norm_type=None, act_type='leakyrelu', mode='CNA', convtype='Conv2D', spectral_norm=False, plus=False, c2x2=False):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus=plus, c2x2=c2x2)
        self.RDB2 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus=plus, c2x2=c2x2)
        self.RDB3 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus=plus, c2x2=c2x2)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class UNet2DConditionWrapper(UNet2DConditionModel):
    """Internal class"""

    def forward(self, sample, timestep, encoder_hidden_states, _=None, __=None, ___=None, ____=None, _____=None, ______=None, _______: 'bool'=True) ->Tuple:
        sample = sample
        timestep = timestep
        encoder_hidden_states = encoder_hidden_states
        sample = UNet2DConditionModel.forward(self, sample, timestep, encoder_hidden_states, return_dict=True).sample
        return sample,


class AutoencoderKLWrapper(AutoencoderKL):
    """Internal class"""

    def encode(self, x) ->Tuple:
        x = x
        outputs: 'AutoencoderKLOutput' = AutoencoderKL.encode(self, x, True)
        return outputs.latent_dist.sample(),

    def decode(self, z) ->Tuple:
        z = z
        outputs: 'DecoderOutput' = AutoencoderKL.decode(self, z, True)
        return outputs.sample,


class DeepDanbooruModel(nn.Module):

    def __init__(self):
        super(DeepDanbooruModel, self).__init__()
        self.n_Conv_0 = nn.Conv2d(kernel_size=(7, 7), in_channels=3, out_channels=64, stride=(2, 2))
        self.n_MaxPool_0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.n_Conv_1 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_2 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=64)
        self.n_Conv_3 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_4 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_5 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=64)
        self.n_Conv_6 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_7 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_8 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=64)
        self.n_Conv_9 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_10 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_11 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=512, stride=(2, 2))
        self.n_Conv_12 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=128)
        self.n_Conv_13 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, stride=(2, 2))
        self.n_Conv_14 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_15 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_16 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_17 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_18 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_19 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_20 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_21 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_22 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_23 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_24 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_25 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_26 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_27 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_28 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_29 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_30 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_31 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_32 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_33 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_34 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_35 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_36 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=1024, stride=(2, 2))
        self.n_Conv_37 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=256)
        self.n_Conv_38 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(2, 2))
        self.n_Conv_39 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_40 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_41 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_42 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_43 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_44 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_45 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_46 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_47 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_48 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_49 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_50 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_51 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_52 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_53 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_54 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_55 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_56 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_57 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_58 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_59 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_60 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_61 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_62 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_63 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_64 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_65 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_66 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_67 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_68 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_69 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_70 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_71 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_72 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_73 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_74 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_75 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_76 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_77 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_78 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_79 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_80 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_81 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_82 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_83 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_84 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_85 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_86 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_87 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_88 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_89 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_90 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_91 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_92 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_93 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_94 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_95 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_96 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_97 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_98 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(2, 2))
        self.n_Conv_99 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_100 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=1024, stride=(2, 2))
        self.n_Conv_101 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_102 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_103 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_104 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_105 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_106 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_107 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_108 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_109 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_110 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_111 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_112 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_113 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_114 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_115 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_116 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_117 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_118 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_119 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_120 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_121 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_122 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_123 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_124 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_125 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_126 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_127 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_128 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_129 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_130 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_131 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_132 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_133 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_134 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_135 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_136 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_137 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_138 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_139 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_140 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_141 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_142 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_143 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_144 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_145 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_146 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_147 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_148 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_149 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_150 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_151 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_152 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_153 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_154 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_155 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_156 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_157 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_158 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=2048, stride=(2, 2))
        self.n_Conv_159 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=512)
        self.n_Conv_160 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512, stride=(2, 2))
        self.n_Conv_161 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_162 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=512)
        self.n_Conv_163 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512)
        self.n_Conv_164 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_165 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=512)
        self.n_Conv_166 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512)
        self.n_Conv_167 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_168 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=4096, stride=(2, 2))
        self.n_Conv_169 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=1024)
        self.n_Conv_170 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, stride=(2, 2))
        self.n_Conv_171 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_172 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=1024)
        self.n_Conv_173 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024)
        self.n_Conv_174 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_175 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=1024)
        self.n_Conv_176 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024)
        self.n_Conv_177 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_178 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=9176, bias=False)

    def forward(self, *inputs):
        t_358, = inputs
        t_359 = t_358.permute(*[0, 3, 1, 2])
        t_359_padded = F.pad(t_359, [2, 3, 2, 3], value=0)
        t_360 = self.n_Conv_0(t_359_padded)
        t_361 = F.relu(t_360)
        t_361 = F.pad(t_361, [0, 1, 0, 1], value=float('-inf'))
        t_362 = self.n_MaxPool_0(t_361)
        t_363 = self.n_Conv_1(t_362)
        t_364 = self.n_Conv_2(t_362)
        t_365 = F.relu(t_364)
        t_365_padded = F.pad(t_365, [1, 1, 1, 1], value=0)
        t_366 = self.n_Conv_3(t_365_padded)
        t_367 = F.relu(t_366)
        t_368 = self.n_Conv_4(t_367)
        t_369 = torch.add(t_368, t_363)
        t_370 = F.relu(t_369)
        t_371 = self.n_Conv_5(t_370)
        t_372 = F.relu(t_371)
        t_372_padded = F.pad(t_372, [1, 1, 1, 1], value=0)
        t_373 = self.n_Conv_6(t_372_padded)
        t_374 = F.relu(t_373)
        t_375 = self.n_Conv_7(t_374)
        t_376 = torch.add(t_375, t_370)
        t_377 = F.relu(t_376)
        t_378 = self.n_Conv_8(t_377)
        t_379 = F.relu(t_378)
        t_379_padded = F.pad(t_379, [1, 1, 1, 1], value=0)
        t_380 = self.n_Conv_9(t_379_padded)
        t_381 = F.relu(t_380)
        t_382 = self.n_Conv_10(t_381)
        t_383 = torch.add(t_382, t_377)
        t_384 = F.relu(t_383)
        t_385 = self.n_Conv_11(t_384)
        t_386 = self.n_Conv_12(t_384)
        t_387 = F.relu(t_386)
        t_387_padded = F.pad(t_387, [0, 1, 0, 1], value=0)
        t_388 = self.n_Conv_13(t_387_padded)
        t_389 = F.relu(t_388)
        t_390 = self.n_Conv_14(t_389)
        t_391 = torch.add(t_390, t_385)
        t_392 = F.relu(t_391)
        t_393 = self.n_Conv_15(t_392)
        t_394 = F.relu(t_393)
        t_394_padded = F.pad(t_394, [1, 1, 1, 1], value=0)
        t_395 = self.n_Conv_16(t_394_padded)
        t_396 = F.relu(t_395)
        t_397 = self.n_Conv_17(t_396)
        t_398 = torch.add(t_397, t_392)
        t_399 = F.relu(t_398)
        t_400 = self.n_Conv_18(t_399)
        t_401 = F.relu(t_400)
        t_401_padded = F.pad(t_401, [1, 1, 1, 1], value=0)
        t_402 = self.n_Conv_19(t_401_padded)
        t_403 = F.relu(t_402)
        t_404 = self.n_Conv_20(t_403)
        t_405 = torch.add(t_404, t_399)
        t_406 = F.relu(t_405)
        t_407 = self.n_Conv_21(t_406)
        t_408 = F.relu(t_407)
        t_408_padded = F.pad(t_408, [1, 1, 1, 1], value=0)
        t_409 = self.n_Conv_22(t_408_padded)
        t_410 = F.relu(t_409)
        t_411 = self.n_Conv_23(t_410)
        t_412 = torch.add(t_411, t_406)
        t_413 = F.relu(t_412)
        t_414 = self.n_Conv_24(t_413)
        t_415 = F.relu(t_414)
        t_415_padded = F.pad(t_415, [1, 1, 1, 1], value=0)
        t_416 = self.n_Conv_25(t_415_padded)
        t_417 = F.relu(t_416)
        t_418 = self.n_Conv_26(t_417)
        t_419 = torch.add(t_418, t_413)
        t_420 = F.relu(t_419)
        t_421 = self.n_Conv_27(t_420)
        t_422 = F.relu(t_421)
        t_422_padded = F.pad(t_422, [1, 1, 1, 1], value=0)
        t_423 = self.n_Conv_28(t_422_padded)
        t_424 = F.relu(t_423)
        t_425 = self.n_Conv_29(t_424)
        t_426 = torch.add(t_425, t_420)
        t_427 = F.relu(t_426)
        t_428 = self.n_Conv_30(t_427)
        t_429 = F.relu(t_428)
        t_429_padded = F.pad(t_429, [1, 1, 1, 1], value=0)
        t_430 = self.n_Conv_31(t_429_padded)
        t_431 = F.relu(t_430)
        t_432 = self.n_Conv_32(t_431)
        t_433 = torch.add(t_432, t_427)
        t_434 = F.relu(t_433)
        t_435 = self.n_Conv_33(t_434)
        t_436 = F.relu(t_435)
        t_436_padded = F.pad(t_436, [1, 1, 1, 1], value=0)
        t_437 = self.n_Conv_34(t_436_padded)
        t_438 = F.relu(t_437)
        t_439 = self.n_Conv_35(t_438)
        t_440 = torch.add(t_439, t_434)
        t_441 = F.relu(t_440)
        t_442 = self.n_Conv_36(t_441)
        t_443 = self.n_Conv_37(t_441)
        t_444 = F.relu(t_443)
        t_444_padded = F.pad(t_444, [0, 1, 0, 1], value=0)
        t_445 = self.n_Conv_38(t_444_padded)
        t_446 = F.relu(t_445)
        t_447 = self.n_Conv_39(t_446)
        t_448 = torch.add(t_447, t_442)
        t_449 = F.relu(t_448)
        t_450 = self.n_Conv_40(t_449)
        t_451 = F.relu(t_450)
        t_451_padded = F.pad(t_451, [1, 1, 1, 1], value=0)
        t_452 = self.n_Conv_41(t_451_padded)
        t_453 = F.relu(t_452)
        t_454 = self.n_Conv_42(t_453)
        t_455 = torch.add(t_454, t_449)
        t_456 = F.relu(t_455)
        t_457 = self.n_Conv_43(t_456)
        t_458 = F.relu(t_457)
        t_458_padded = F.pad(t_458, [1, 1, 1, 1], value=0)
        t_459 = self.n_Conv_44(t_458_padded)
        t_460 = F.relu(t_459)
        t_461 = self.n_Conv_45(t_460)
        t_462 = torch.add(t_461, t_456)
        t_463 = F.relu(t_462)
        t_464 = self.n_Conv_46(t_463)
        t_465 = F.relu(t_464)
        t_465_padded = F.pad(t_465, [1, 1, 1, 1], value=0)
        t_466 = self.n_Conv_47(t_465_padded)
        t_467 = F.relu(t_466)
        t_468 = self.n_Conv_48(t_467)
        t_469 = torch.add(t_468, t_463)
        t_470 = F.relu(t_469)
        t_471 = self.n_Conv_49(t_470)
        t_472 = F.relu(t_471)
        t_472_padded = F.pad(t_472, [1, 1, 1, 1], value=0)
        t_473 = self.n_Conv_50(t_472_padded)
        t_474 = F.relu(t_473)
        t_475 = self.n_Conv_51(t_474)
        t_476 = torch.add(t_475, t_470)
        t_477 = F.relu(t_476)
        t_478 = self.n_Conv_52(t_477)
        t_479 = F.relu(t_478)
        t_479_padded = F.pad(t_479, [1, 1, 1, 1], value=0)
        t_480 = self.n_Conv_53(t_479_padded)
        t_481 = F.relu(t_480)
        t_482 = self.n_Conv_54(t_481)
        t_483 = torch.add(t_482, t_477)
        t_484 = F.relu(t_483)
        t_485 = self.n_Conv_55(t_484)
        t_486 = F.relu(t_485)
        t_486_padded = F.pad(t_486, [1, 1, 1, 1], value=0)
        t_487 = self.n_Conv_56(t_486_padded)
        t_488 = F.relu(t_487)
        t_489 = self.n_Conv_57(t_488)
        t_490 = torch.add(t_489, t_484)
        t_491 = F.relu(t_490)
        t_492 = self.n_Conv_58(t_491)
        t_493 = F.relu(t_492)
        t_493_padded = F.pad(t_493, [1, 1, 1, 1], value=0)
        t_494 = self.n_Conv_59(t_493_padded)
        t_495 = F.relu(t_494)
        t_496 = self.n_Conv_60(t_495)
        t_497 = torch.add(t_496, t_491)
        t_498 = F.relu(t_497)
        t_499 = self.n_Conv_61(t_498)
        t_500 = F.relu(t_499)
        t_500_padded = F.pad(t_500, [1, 1, 1, 1], value=0)
        t_501 = self.n_Conv_62(t_500_padded)
        t_502 = F.relu(t_501)
        t_503 = self.n_Conv_63(t_502)
        t_504 = torch.add(t_503, t_498)
        t_505 = F.relu(t_504)
        t_506 = self.n_Conv_64(t_505)
        t_507 = F.relu(t_506)
        t_507_padded = F.pad(t_507, [1, 1, 1, 1], value=0)
        t_508 = self.n_Conv_65(t_507_padded)
        t_509 = F.relu(t_508)
        t_510 = self.n_Conv_66(t_509)
        t_511 = torch.add(t_510, t_505)
        t_512 = F.relu(t_511)
        t_513 = self.n_Conv_67(t_512)
        t_514 = F.relu(t_513)
        t_514_padded = F.pad(t_514, [1, 1, 1, 1], value=0)
        t_515 = self.n_Conv_68(t_514_padded)
        t_516 = F.relu(t_515)
        t_517 = self.n_Conv_69(t_516)
        t_518 = torch.add(t_517, t_512)
        t_519 = F.relu(t_518)
        t_520 = self.n_Conv_70(t_519)
        t_521 = F.relu(t_520)
        t_521_padded = F.pad(t_521, [1, 1, 1, 1], value=0)
        t_522 = self.n_Conv_71(t_521_padded)
        t_523 = F.relu(t_522)
        t_524 = self.n_Conv_72(t_523)
        t_525 = torch.add(t_524, t_519)
        t_526 = F.relu(t_525)
        t_527 = self.n_Conv_73(t_526)
        t_528 = F.relu(t_527)
        t_528_padded = F.pad(t_528, [1, 1, 1, 1], value=0)
        t_529 = self.n_Conv_74(t_528_padded)
        t_530 = F.relu(t_529)
        t_531 = self.n_Conv_75(t_530)
        t_532 = torch.add(t_531, t_526)
        t_533 = F.relu(t_532)
        t_534 = self.n_Conv_76(t_533)
        t_535 = F.relu(t_534)
        t_535_padded = F.pad(t_535, [1, 1, 1, 1], value=0)
        t_536 = self.n_Conv_77(t_535_padded)
        t_537 = F.relu(t_536)
        t_538 = self.n_Conv_78(t_537)
        t_539 = torch.add(t_538, t_533)
        t_540 = F.relu(t_539)
        t_541 = self.n_Conv_79(t_540)
        t_542 = F.relu(t_541)
        t_542_padded = F.pad(t_542, [1, 1, 1, 1], value=0)
        t_543 = self.n_Conv_80(t_542_padded)
        t_544 = F.relu(t_543)
        t_545 = self.n_Conv_81(t_544)
        t_546 = torch.add(t_545, t_540)
        t_547 = F.relu(t_546)
        t_548 = self.n_Conv_82(t_547)
        t_549 = F.relu(t_548)
        t_549_padded = F.pad(t_549, [1, 1, 1, 1], value=0)
        t_550 = self.n_Conv_83(t_549_padded)
        t_551 = F.relu(t_550)
        t_552 = self.n_Conv_84(t_551)
        t_553 = torch.add(t_552, t_547)
        t_554 = F.relu(t_553)
        t_555 = self.n_Conv_85(t_554)
        t_556 = F.relu(t_555)
        t_556_padded = F.pad(t_556, [1, 1, 1, 1], value=0)
        t_557 = self.n_Conv_86(t_556_padded)
        t_558 = F.relu(t_557)
        t_559 = self.n_Conv_87(t_558)
        t_560 = torch.add(t_559, t_554)
        t_561 = F.relu(t_560)
        t_562 = self.n_Conv_88(t_561)
        t_563 = F.relu(t_562)
        t_563_padded = F.pad(t_563, [1, 1, 1, 1], value=0)
        t_564 = self.n_Conv_89(t_563_padded)
        t_565 = F.relu(t_564)
        t_566 = self.n_Conv_90(t_565)
        t_567 = torch.add(t_566, t_561)
        t_568 = F.relu(t_567)
        t_569 = self.n_Conv_91(t_568)
        t_570 = F.relu(t_569)
        t_570_padded = F.pad(t_570, [1, 1, 1, 1], value=0)
        t_571 = self.n_Conv_92(t_570_padded)
        t_572 = F.relu(t_571)
        t_573 = self.n_Conv_93(t_572)
        t_574 = torch.add(t_573, t_568)
        t_575 = F.relu(t_574)
        t_576 = self.n_Conv_94(t_575)
        t_577 = F.relu(t_576)
        t_577_padded = F.pad(t_577, [1, 1, 1, 1], value=0)
        t_578 = self.n_Conv_95(t_577_padded)
        t_579 = F.relu(t_578)
        t_580 = self.n_Conv_96(t_579)
        t_581 = torch.add(t_580, t_575)
        t_582 = F.relu(t_581)
        t_583 = self.n_Conv_97(t_582)
        t_584 = F.relu(t_583)
        t_584_padded = F.pad(t_584, [0, 1, 0, 1], value=0)
        t_585 = self.n_Conv_98(t_584_padded)
        t_586 = F.relu(t_585)
        t_587 = self.n_Conv_99(t_586)
        t_588 = self.n_Conv_100(t_582)
        t_589 = torch.add(t_587, t_588)
        t_590 = F.relu(t_589)
        t_591 = self.n_Conv_101(t_590)
        t_592 = F.relu(t_591)
        t_592_padded = F.pad(t_592, [1, 1, 1, 1], value=0)
        t_593 = self.n_Conv_102(t_592_padded)
        t_594 = F.relu(t_593)
        t_595 = self.n_Conv_103(t_594)
        t_596 = torch.add(t_595, t_590)
        t_597 = F.relu(t_596)
        t_598 = self.n_Conv_104(t_597)
        t_599 = F.relu(t_598)
        t_599_padded = F.pad(t_599, [1, 1, 1, 1], value=0)
        t_600 = self.n_Conv_105(t_599_padded)
        t_601 = F.relu(t_600)
        t_602 = self.n_Conv_106(t_601)
        t_603 = torch.add(t_602, t_597)
        t_604 = F.relu(t_603)
        t_605 = self.n_Conv_107(t_604)
        t_606 = F.relu(t_605)
        t_606_padded = F.pad(t_606, [1, 1, 1, 1], value=0)
        t_607 = self.n_Conv_108(t_606_padded)
        t_608 = F.relu(t_607)
        t_609 = self.n_Conv_109(t_608)
        t_610 = torch.add(t_609, t_604)
        t_611 = F.relu(t_610)
        t_612 = self.n_Conv_110(t_611)
        t_613 = F.relu(t_612)
        t_613_padded = F.pad(t_613, [1, 1, 1, 1], value=0)
        t_614 = self.n_Conv_111(t_613_padded)
        t_615 = F.relu(t_614)
        t_616 = self.n_Conv_112(t_615)
        t_617 = torch.add(t_616, t_611)
        t_618 = F.relu(t_617)
        t_619 = self.n_Conv_113(t_618)
        t_620 = F.relu(t_619)
        t_620_padded = F.pad(t_620, [1, 1, 1, 1], value=0)
        t_621 = self.n_Conv_114(t_620_padded)
        t_622 = F.relu(t_621)
        t_623 = self.n_Conv_115(t_622)
        t_624 = torch.add(t_623, t_618)
        t_625 = F.relu(t_624)
        t_626 = self.n_Conv_116(t_625)
        t_627 = F.relu(t_626)
        t_627_padded = F.pad(t_627, [1, 1, 1, 1], value=0)
        t_628 = self.n_Conv_117(t_627_padded)
        t_629 = F.relu(t_628)
        t_630 = self.n_Conv_118(t_629)
        t_631 = torch.add(t_630, t_625)
        t_632 = F.relu(t_631)
        t_633 = self.n_Conv_119(t_632)
        t_634 = F.relu(t_633)
        t_634_padded = F.pad(t_634, [1, 1, 1, 1], value=0)
        t_635 = self.n_Conv_120(t_634_padded)
        t_636 = F.relu(t_635)
        t_637 = self.n_Conv_121(t_636)
        t_638 = torch.add(t_637, t_632)
        t_639 = F.relu(t_638)
        t_640 = self.n_Conv_122(t_639)
        t_641 = F.relu(t_640)
        t_641_padded = F.pad(t_641, [1, 1, 1, 1], value=0)
        t_642 = self.n_Conv_123(t_641_padded)
        t_643 = F.relu(t_642)
        t_644 = self.n_Conv_124(t_643)
        t_645 = torch.add(t_644, t_639)
        t_646 = F.relu(t_645)
        t_647 = self.n_Conv_125(t_646)
        t_648 = F.relu(t_647)
        t_648_padded = F.pad(t_648, [1, 1, 1, 1], value=0)
        t_649 = self.n_Conv_126(t_648_padded)
        t_650 = F.relu(t_649)
        t_651 = self.n_Conv_127(t_650)
        t_652 = torch.add(t_651, t_646)
        t_653 = F.relu(t_652)
        t_654 = self.n_Conv_128(t_653)
        t_655 = F.relu(t_654)
        t_655_padded = F.pad(t_655, [1, 1, 1, 1], value=0)
        t_656 = self.n_Conv_129(t_655_padded)
        t_657 = F.relu(t_656)
        t_658 = self.n_Conv_130(t_657)
        t_659 = torch.add(t_658, t_653)
        t_660 = F.relu(t_659)
        t_661 = self.n_Conv_131(t_660)
        t_662 = F.relu(t_661)
        t_662_padded = F.pad(t_662, [1, 1, 1, 1], value=0)
        t_663 = self.n_Conv_132(t_662_padded)
        t_664 = F.relu(t_663)
        t_665 = self.n_Conv_133(t_664)
        t_666 = torch.add(t_665, t_660)
        t_667 = F.relu(t_666)
        t_668 = self.n_Conv_134(t_667)
        t_669 = F.relu(t_668)
        t_669_padded = F.pad(t_669, [1, 1, 1, 1], value=0)
        t_670 = self.n_Conv_135(t_669_padded)
        t_671 = F.relu(t_670)
        t_672 = self.n_Conv_136(t_671)
        t_673 = torch.add(t_672, t_667)
        t_674 = F.relu(t_673)
        t_675 = self.n_Conv_137(t_674)
        t_676 = F.relu(t_675)
        t_676_padded = F.pad(t_676, [1, 1, 1, 1], value=0)
        t_677 = self.n_Conv_138(t_676_padded)
        t_678 = F.relu(t_677)
        t_679 = self.n_Conv_139(t_678)
        t_680 = torch.add(t_679, t_674)
        t_681 = F.relu(t_680)
        t_682 = self.n_Conv_140(t_681)
        t_683 = F.relu(t_682)
        t_683_padded = F.pad(t_683, [1, 1, 1, 1], value=0)
        t_684 = self.n_Conv_141(t_683_padded)
        t_685 = F.relu(t_684)
        t_686 = self.n_Conv_142(t_685)
        t_687 = torch.add(t_686, t_681)
        t_688 = F.relu(t_687)
        t_689 = self.n_Conv_143(t_688)
        t_690 = F.relu(t_689)
        t_690_padded = F.pad(t_690, [1, 1, 1, 1], value=0)
        t_691 = self.n_Conv_144(t_690_padded)
        t_692 = F.relu(t_691)
        t_693 = self.n_Conv_145(t_692)
        t_694 = torch.add(t_693, t_688)
        t_695 = F.relu(t_694)
        t_696 = self.n_Conv_146(t_695)
        t_697 = F.relu(t_696)
        t_697_padded = F.pad(t_697, [1, 1, 1, 1], value=0)
        t_698 = self.n_Conv_147(t_697_padded)
        t_699 = F.relu(t_698)
        t_700 = self.n_Conv_148(t_699)
        t_701 = torch.add(t_700, t_695)
        t_702 = F.relu(t_701)
        t_703 = self.n_Conv_149(t_702)
        t_704 = F.relu(t_703)
        t_704_padded = F.pad(t_704, [1, 1, 1, 1], value=0)
        t_705 = self.n_Conv_150(t_704_padded)
        t_706 = F.relu(t_705)
        t_707 = self.n_Conv_151(t_706)
        t_708 = torch.add(t_707, t_702)
        t_709 = F.relu(t_708)
        t_710 = self.n_Conv_152(t_709)
        t_711 = F.relu(t_710)
        t_711_padded = F.pad(t_711, [1, 1, 1, 1], value=0)
        t_712 = self.n_Conv_153(t_711_padded)
        t_713 = F.relu(t_712)
        t_714 = self.n_Conv_154(t_713)
        t_715 = torch.add(t_714, t_709)
        t_716 = F.relu(t_715)
        t_717 = self.n_Conv_155(t_716)
        t_718 = F.relu(t_717)
        t_718_padded = F.pad(t_718, [1, 1, 1, 1], value=0)
        t_719 = self.n_Conv_156(t_718_padded)
        t_720 = F.relu(t_719)
        t_721 = self.n_Conv_157(t_720)
        t_722 = torch.add(t_721, t_716)
        t_723 = F.relu(t_722)
        t_724 = self.n_Conv_158(t_723)
        t_725 = self.n_Conv_159(t_723)
        t_726 = F.relu(t_725)
        t_726_padded = F.pad(t_726, [0, 1, 0, 1], value=0)
        t_727 = self.n_Conv_160(t_726_padded)
        t_728 = F.relu(t_727)
        t_729 = self.n_Conv_161(t_728)
        t_730 = torch.add(t_729, t_724)
        t_731 = F.relu(t_730)
        t_732 = self.n_Conv_162(t_731)
        t_733 = F.relu(t_732)
        t_733_padded = F.pad(t_733, [1, 1, 1, 1], value=0)
        t_734 = self.n_Conv_163(t_733_padded)
        t_735 = F.relu(t_734)
        t_736 = self.n_Conv_164(t_735)
        t_737 = torch.add(t_736, t_731)
        t_738 = F.relu(t_737)
        t_739 = self.n_Conv_165(t_738)
        t_740 = F.relu(t_739)
        t_740_padded = F.pad(t_740, [1, 1, 1, 1], value=0)
        t_741 = self.n_Conv_166(t_740_padded)
        t_742 = F.relu(t_741)
        t_743 = self.n_Conv_167(t_742)
        t_744 = torch.add(t_743, t_738)
        t_745 = F.relu(t_744)
        t_746 = self.n_Conv_168(t_745)
        t_747 = self.n_Conv_169(t_745)
        t_748 = F.relu(t_747)
        t_748_padded = F.pad(t_748, [0, 1, 0, 1], value=0)
        t_749 = self.n_Conv_170(t_748_padded)
        t_750 = F.relu(t_749)
        t_751 = self.n_Conv_171(t_750)
        t_752 = torch.add(t_751, t_746)
        t_753 = F.relu(t_752)
        t_754 = self.n_Conv_172(t_753)
        t_755 = F.relu(t_754)
        t_755_padded = F.pad(t_755, [1, 1, 1, 1], value=0)
        t_756 = self.n_Conv_173(t_755_padded)
        t_757 = F.relu(t_756)
        t_758 = self.n_Conv_174(t_757)
        t_759 = torch.add(t_758, t_753)
        t_760 = F.relu(t_759)
        t_761 = self.n_Conv_175(t_760)
        t_762 = F.relu(t_761)
        t_762_padded = F.pad(t_762, [1, 1, 1, 1], value=0)
        t_763 = self.n_Conv_176(t_762_padded)
        t_764 = F.relu(t_763)
        t_765 = self.n_Conv_177(t_764)
        t_766 = torch.add(t_765, t_760)
        t_767 = F.relu(t_766)
        t_768 = self.n_Conv_178(t_767)
        t_769 = F.avg_pool2d(t_768, kernel_size=(4, 4))
        t_770 = torch.squeeze(t_769, 3)
        t_770 = torch.squeeze(t_770, 2)
        t_771 = torch.sigmoid(t_770)
        return t_771

    def load_state_dict(self, state_dict, **kwargs):
        super(DeepDanbooruModel, self).load_state_dict({k: v for k, v in state_dict.items() if k != 'tags'})


class MultiheadAttention(torch.nn.MultiheadAttention):
    """Normal torch multihead attention. Taken once again from @Birch-sans diffusers-play repository. Thank you <3"""

    def __init__(self, query_dim: 'int', cross_attention_dim: 'Optional[int]'=None, heads: 'int'=8, dim_head: 'int'=64, dropout: 'float'=0.0, bias=False):
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        super().__init__(embed_dim=inner_dim, num_heads=heads, dropout=dropout, bias=bias, batch_first=True, kdim=cross_attention_dim, vdim=cross_attention_dim)

    def forward(self, hidden_states: 'torch.Tensor', encoder_hidden_states: 'Optional[torch.Tensor]'=None, attention_mask: 'Optional[torch.Tensor]'=None, **_) ->torch.Tensor:
        kv = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
            _, vision_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, vision_tokens, -1)
        out, _ = super().forward(query=hidden_states, key=kv, value=kv, need_weights=False, attn_mask=attention_mask)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConcatBlock,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample2D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Get_gradient_nopadding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RRDB,
     lambda: ([], {'nf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNetBlock,
     lambda: ([], {'in_nc': 4, 'mid_nc': 4, 'out_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualDenseBlock_5C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     False),
    (ShortcutBlock,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShortcutBlockSPSR,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_VoltaML_voltaML_fast_stable_diffusion(_paritybench_base):
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

