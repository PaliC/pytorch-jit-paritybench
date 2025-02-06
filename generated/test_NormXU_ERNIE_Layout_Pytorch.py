import sys
_module = sys.modules[__name__]
del sys
compare_output = _module
test_ernie_qa = _module
test_ernie_token_cls = _module
networks = _module
configuration_erine_layout = _module
configuration_extrapolation = _module
modeling_erine_layout = _module
modeling_erine_layout_extrapolation = _module
processing_ernie_layout = _module
tokenization_ernie_layout_fast = _module
visual_backbone = _module
test_flash_attn = _module
convert2torch = _module

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


import numpy as np


import torch


import torch.nn.functional as F


import math


import torch.nn as nn


from torch.nn import CrossEntropyLoss


from collections import OrderedDict


class ErnieLayoutPooler(nn.Module):

    def __init__(self, hidden_size, with_pool):
        super(ErnieLayoutPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.with_pool = with_pool

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.with_pool == 'tanh':
            pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieLayoutEmbeddings(nn.Module):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super(ErnieLayoutEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _calc_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError('The `bbox` coordinate values should be within 0-1000 range.') from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        return left_position_embeddings, upper_position_embeddings, right_position_embeddings, lower_position_embeddings, h_position_embeddings, w_position_embeddings

    def forward(self, input_ids, bbox=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        words_embeddings = inputs_embeds
        if position_ids is None:
            ones = torch.ones_like(input_ids)
            seq_length = torch.cumsum(ones, dim=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        position_embeddings = self.position_embeddings(position_ids)
        x1, y1, x2, y2, h, w = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + x1 + y1 + x2 + y2 + h + w + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
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
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) ->str:
        return 'p={}'.format(self.drop_prob)


class ErnieLayoutSelfOutput(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.drop_path_rate = 0.0 if not hasattr(config, 'hidden_dropout_prob') else config.hidden_dropout_prob
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def exists(val):
    return val is not None


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = -dim - 1 if dim < 0 else t.ndim - dim - 1
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


class AlibiPositionalBias(nn.Module):

    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):

        def get_slopes_power_of_2(n):
            start = 2 ** -2 ** -(math.log2(n) - 3)
            ratio = start
            return [(start * ratio ** i) for i in range(n)]
        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)
        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device
        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., :i, :j]
        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer('bias', bias, persistent=False)
        return self.bias


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        self.register_buffer('inv_freq', inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.cos_cached.size()[2]:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """
        copied from LLamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
        'fix_based' is inspired from  https://normxu.github.io/Rethinking-Rotary-Position-Embedding-2/
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, fix_base=False):
        self.scaling_factor = scaling_factor
        self.fix_base = fix_base
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len > self.max_position_embeddings:
            lamda_factor = (self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
            base = self.base * lamda_factor
            inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2).float() / self.dim)
            if self.fix_base:
                inv_freq = inv_freq * 1.0 / lamda_factor
            self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)


class LearnedAlibiPositionalBias(AlibiPositionalBias):

    def __init__(self, heads, total_heads):
        super().__init__(heads, total_heads)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def forward(self, i, j):
        h, device = self.heads, self.device

        def get_slopes(param):
            return pad_at_dim(param.exp(), (0, h - param.shape[0]), dim=-2)
        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            bias = self.bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, device)
            self.register_buffer('bias', bias, persistent=False)
        slopes = get_slopes(self.learned_logslopes)
        bias = bias * slopes
        return bias


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """Copied from LLamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)


class MixedNTKScalingRotaryEmbedding(RotaryEmbedding):
    """
        copied from LLamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
        'fix_based' is inspired from  https://normxu.github.io/Rethinking-Rotary-Position-Embedding-2/
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, b=0.75, device=None):
        self.b = b
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len > self.max_position_embeddings:
            k = seq_len / self.max_position_embeddings
            a = np.log(k) / (self.dim // 2 ** self.b)
            inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
            inv_freq /= (a * torch.arange(1, self.dim // 2 + 1).float() ** self.b).exp()
            self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, seq_len):
    position_ids = torch.arange(seq_len).expand((1, -1))
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


logger = logging.get_logger(__name__)


class ErnieLayoutSelfAttention(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout_p = config.attention_probs_dropout_prob
        self.use_flash_attn = getattr(config, 'use_flash_attn', False)
        if not self.use_flash_attn:
            self.dropout = nn.Dropout(self.dropout_p)
        self.max_position_embeddings = config.max_position_embeddings
        self.use_entropy_scale = config.use_entropy_scale
        self.use_alibi = config.use_alibi
        self.use_rope_attention_bias = config.use_rope_attention_bias
        if config.use_alibi:
            alibi_num_heads = config.alibi_num_heads if hasattr(config, 'alibi_num_heads') else self.num_attention_heads
            assert alibi_num_heads <= self.num_attention_heads, 'number of ALiBi heads must be less than the total number of heads'
            if config.learnable_alibi:
                self.alibi = LearnedAlibiPositionalBias(heads=alibi_num_heads, total_heads=self.num_attention_heads)
            else:
                self.alibi = AlibiPositionalBias(heads=alibi_num_heads, total_heads=self.num_attention_heads)
        if config.use_rope_attention_bias:
            max_position_embeddings = config.max_position_embeddings
            if config.consequent_visual_bias:
                max_position_embeddings += config.image_feature_pool_shape[0] * config.image_feature_pool_shape[1]
            else:
                visual_max_position_embeddings = config.image_feature_pool_shape[0] * config.image_feature_pool_shape[1]
                self.visual_rotary_emb = self._init_rope(config, visual_max_position_embeddings)
            self.rotary_emb = self._init_rope(config, max_position_embeddings)

    def _init_rope(self, config, max_position_embeddings):
        if config.rope_scaling_factor is None:
            rotary_emb = RotaryEmbedding(self.attention_head_size, max_position_embeddings=max_position_embeddings)
        else:
            scaling_type = config.rope_type
            scaling_factor = config.rope_scaling_factor
            if scaling_type == 'linear':
                rotary_emb = LinearScalingRotaryEmbedding(self.attention_head_size, max_position_embeddings=max_position_embeddings, scaling_factor=scaling_factor)
            elif scaling_type == 'dynamic':
                rotary_emb = DynamicNTKScalingRotaryEmbedding(self.attention_head_size, max_position_embeddings=max_position_embeddings, scaling_factor=scaling_factor, fix_base=config.fix_base)
            elif scaling_type == 'mixed_base':
                rotary_emb = MixedNTKScalingRotaryEmbedding(self.attention_head_size, max_position_embeddings=max_position_embeddings, b=config.b)
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')
        return rotary_emb

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        return q, k, v

    def forward(self, hidden_states, attention_mask=None, head_mask=None, past_key_value=None, output_attentions=False, rel_pos=None, rel_2d_pos=None):
        q, k, v = self.compute_qkv(hidden_states)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)
        bz, num_head, seq_len, _ = key_layer.size()
        if self.use_rope_attention_bias:
            if not self.config.consequent_visual_bias:
                visual_seq_len = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
                cos_visual, sin_visual = self.visual_rotary_emb(value_layer, seq_len=visual_seq_len)
                cos_text, sin_text = self.rotary_emb(value_layer, seq_len=seq_len - visual_seq_len)
                cos = torch.cat([cos_text, cos_visual], dim=-2)
                sin = torch.cat([sin_text, sin_visual], dim=-2)
            else:
                cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, seq_len)
        if self.use_entropy_scale:
            query_layer *= ((torch.arange(0, seq_len) + 1)[None, None, :, None].log() / np.log(self.max_position_embeddings)).clip(1)
        if self.use_flash_attn:
            logger.info('use flash attention')
            attention_probs = None
            bz, _, seq_len, _ = key_layer.size()
            attn_bias = torch.zeros((bz, 1, seq_len, seq_len), dtype=key_layer.dtype, device=key_layer.device)
            attn_bias.masked_fill_(attention_mask, float('-inf'))
            if self.has_relative_attention_bias:
                attn_bias = attn_bias + rel_pos
            if self.has_spatial_attention_bias:
                attn_bias = attn_bias + rel_2d_pos
            if self.use_alibi:
                i, j = map(lambda t: t.shape[-2], (q, k))
                attn_bias = attn_bias + self.alibi(i, j)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True):
                context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, dropout_p=self.dropout_p if self.training else 0.0, attn_mask=attn_bias)
        else:
            query_layer = query_layer / math.sqrt(self.attention_head_size)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if self.has_relative_attention_bias:
                attention_scores += rel_pos
            if self.has_spatial_attention_bias:
                attention_scores += rel_2d_pos
            if self.use_alibi:
                i, j = map(lambda t: t.shape[-2], (q, k))
                attention_scores += self.alibi(i, j)
            attention_scores = attention_scores.float().masked_fill_(attention_mask, torch.finfo(attention_scores.dtype).min)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
            attention_probs = self.dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ErnieLayoutAttention(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutAttention, self).__init__()
        self.self = ErnieLayoutSelfAttention(config)
        self.output = ErnieLayoutSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, past_key_value=None, output_attentions=False, rel_pos=None, rel_2d_pos=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, past_key_value, output_attentions, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)
        attention_output = self.output(self_outputs[0], hidden_states)
        if output_attentions:
            outputs = [attention_output] + self_outputs[1:]
        else:
            outputs = [attention_output]
        return outputs


class ErnieLayoutIntermediate(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == 'gelu':
            self.intermediate_act_fn = nn.GELU()
        else:
            assert False, 'hidden_act is set as: {}, please check it..'.format(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ErnieLayoutOutput(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.drop_path_rate = 0.0 if not hasattr(config, 'drop_path_rate') else config.drop_path_rate
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ErnieLayoutLayer(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutLayer, self).__init__()
        self.seq_len_dim = 1
        self.attention = ErnieLayoutAttention(config)
        self.add_cross_attention = False
        self.intermediate = ErnieLayoutIntermediate(config)
        self.output = ErnieLayoutOutput(config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(self, hidden_states, attention_mask=None, head_mask=None, past_key_value=None, output_attentions=False, rel_pos=None, rel_2d_pos=None):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)
        attention_output = self_attention_outputs[0]
        layer_output = self.feed_forward_chunk(attention_output)
        if output_attentions:
            outputs = self_attention_outputs[1:]
            outputs = [layer_output] + list(outputs)
        else:
            outputs = [layer_output]
        return outputs


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    val_if_large = torch.minimum(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


warning_message = 'You may select either Alibi positional bias or T5 relative positional bias or RoPE, but please refrain from choosing more than one option simultaneously'


class ErnieLayoutEncoder(nn.Module):

    def __init__(self, config):
        super(ErnieLayoutEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([ErnieLayoutLayer(config) for _ in range(config.num_hidden_layers)])
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        assert not (self.has_relative_attention_bias and (self.config.use_alibi or self.config.use_rope_attention_bias)), warning_message
        assert not (self.config.use_rope_attention_bias and self.config.use_alibi), warning_message
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)
        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(rel_pos_x_2d_mat, num_buckets=self.rel_2d_pos_bins, max_distance=self.max_rel_2d_pos)
        rel_pos_y = relative_position_bucket(rel_pos_y_2d_mat, num_buckets=self.rel_2d_pos_bins, max_distance=self.max_rel_2d_pos)
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(self, hidden_states, attention_mask=None, head_mask=None, past_key_values=None, output_attentions=False, output_hidden_states=False, bbox=None, position_ids=None):
        all_hidden_states = () if output_hidden_states else None
        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None
        hidden_save = dict()
        hidden_save['input_hidden_states'] = hidden_states
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_save['input_attention_mask'] = attention_mask
            hidden_save['input_layer_head_mask'] = layer_head_mask
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, past_key_value, output_attentions, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)
            hidden_states = layer_outputs[0]
            hidden_save['{}_data'.format(i)] = hidden_states
        return hidden_states,


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, shortcut=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.shortcut = shortcut
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, ResBlock, layer_list, num_channels=3, num_filters=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        if num_filters is None:
            num_filters = [64, 128, 256, 512]
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet = nn.Sequential(OrderedDict([]))
        for block in range(len(layer_list)):
            name = f'layer{block}'
            self.resnet.add_module(name, self._make_layer(ResBlock, layer_list[block], planes=num_filters[block], stride=1 if block == 0 else 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.resnet(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        shortcut = None
        layers = []
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            shortcut = nn.Sequential(nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * ResBlock.expansion))
        layers.append(ResBlock(self.in_channels, planes, shortcut=shortcut, stride=stride))
        self.in_channels = planes * ResBlock.expansion
        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)


def ResNetCustomized(layers, channels=3):
    supported_layers = [18, 34, 50, 101, 152]
    assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(supported_layers, layers)
    if layers == 18:
        depth = [2, 2, 2, 2]
    elif layers == 34 or layers == 50:
        depth = [3, 4, 6, 3]
    elif layers == 101:
        depth = [3]
    else:
        depth = [3, 8, 36, 3]
    num_filters = [64, 128, 256, 512]
    return ResNet(Bottleneck, depth, channels, num_filters)


class VisualBackbone(nn.Module):

    def __init__(self, config):
        super(VisualBackbone, self).__init__()
        self.backbone = ResNetCustomized(layers=101)
        self.register_buffer('pixel_mean', torch.tensor([103.53, 116.28, 123.675]).reshape([3, 1, 1]))
        self.register_buffer('pixel_std', torch.tensor([57.375, 57.12, 58.395]).reshape([3, 1, 1]))
        self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])

    def forward(self, images):
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features


class ErnieLayoutPredictionHead(nn.Module):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(ErnieLayoutPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(shape=[vocab_size, hidden_size], dtype=self.transform.weight.dtype, is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = torch.reshape(hidden_states, [-1, hidden_states.shape[-1]])
            hidden_states = hidden_states.gather(0, masked_positions)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.decoder_weight) + self.decoder_bias
        return hidden_states


class ErnieLayoutPretrainingHeads(nn.Module):

    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(ErnieLayoutPretrainingHeads, self).__init__()
        self.predictions = ErnieLayoutPredictionHead(hidden_size, vocab_size, activation, embedding_weights)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, shortcut=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        None
        None
        x += identity
        x = self.relu(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ErnieLayoutOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5, drop_path_rate=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ErnieLayoutPooler,
     lambda: ([], {'hidden_size': 4, 'with_pool': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ErnieLayoutSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (VisualBackbone,
     lambda: ([], {'config': _mock_config(image_feature_pool_shape=[4, 4])}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
]

class Test_NormXU_ERNIE_Layout_Pytorch(_paritybench_base):
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

