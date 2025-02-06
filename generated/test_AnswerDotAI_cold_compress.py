import sys
_module = sys.modules[__name__]
del sys
GPTQ = _module
attention_utils = _module
cache = _module
attention_loss = _module
blogpost_perf = _module
eval = _module
eval_multi = _module
generate = _module
generation_utils = _module
metric = _module
model = _module
parallelize_evals = _module
prompt_compression = _module
quantization_utils = _module
quantize = _module
convert_hf_checkpoint = _module
download = _module
setup = _module
task = _module
tokenizer = _module
tp = _module

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


import torch.fx as fx


import torch.nn as nn


import torch.nn.functional as F


from torch.utils._pytree import tree_flatten


from torch.utils._pytree import tree_unflatten


import math


from typing import Tuple


from torch.nn import functional as F


from abc import ABC


from abc import abstractmethod


from collections import Counter


import time


import itertools


import pandas as pd


import numpy as np


from typing import Optional


from typing import List


from collections import defaultdict


import torch._dynamo.config


import torch._inductor.config


import logging


from torch.nn.attention import SDPBackend


from torch.nn.attention import sdpa_kernel


from typing import Dict


from typing import Any


from torch import Tensor


import re


import string


from typing import Literal


from typing import TypedDict


import torch.distributed as dist


from torch import nn


def unpack_low_bit_tensor(packed_tensor, n_bit, original_shape):
    assert n_bit in [2, 4], 'Only 2-bit and 4-bit unpacking are supported'
    mask = (1 << n_bit) - 1
    original_numel = torch.prod(torch.tensor(original_shape))
    shifts = torch.arange(0, 8, n_bit, device=packed_tensor.device)
    unpacked = (packed_tensor.unsqueeze(1) >> shifts & mask).flatten()
    original = unpacked.reshape(-1)[:original_numel]
    original = original.reshape(original_shape)
    return original


def dequantize_tensor(x, scales, zeros, orig_shape, n_bit=8, axis=0):
    assert n_bit in [2, 4, 8], 'Only 2-bit, 4-bit, and 8-bit quantization are supported'
    if n_bit < 8:
        x = unpack_low_bit_tensor(x, n_bit, orig_shape)
    x = x.transpose(0, axis)
    return x.sub(2 ** (n_bit - 1)).mul(scales.reshape(-1, *([1] * (x.dim() - 1)))).add(zeros.reshape(-1, *([1] * (x.dim() - 1)))).reshape_as(x).transpose(0, axis)


def pack_low_bit_tensor(tensor, n_bit):
    assert n_bit in [2, 4], 'Only 2-bit and 4-bit packing are supported'
    if n_bit == 4:
        assert torch.all(tensor < 16) and torch.all(tensor >= 0), 'All values must be in [0, 15] range for 4-bit packing'
    else:
        assert torch.all(tensor < 4) and torch.all(tensor >= 0), 'All values must be in [0, 3] range for 2-bit packing'
    values_per_byte = 8 // n_bit
    flat_tensor = tensor.flatten()
    if flat_tensor.numel() % values_per_byte != 0:
        padding_size = values_per_byte - flat_tensor.numel() % values_per_byte
        flat_tensor = torch.cat([flat_tensor, flat_tensor.new_zeros(padding_size)])
    reshaped = flat_tensor.reshape(-1, values_per_byte)
    shifts = torch.arange(0, 8, n_bit, device=tensor.device)
    packed = (reshaped << shifts).sum(dim=1).byte()
    return packed


def quantize_tensor(x, n_bit=8, axis=0):
    assert n_bit in [2, 4, 8], 'Only 2-bit, 4-bit, and 8-bit quantization are supported'
    x = x.transpose(0, axis)
    min_val, max_val = torch.aminmax(x.reshape(x.shape[0], -1), dim=1)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-06) / max_int
    zeros = min_val + scales * 2 ** (n_bit - 1)
    x_int8 = x.sub(min_val.reshape(-1, *([1] * (x.dim() - 1)))).div(scales.reshape(-1, *([1] * (x.dim() - 1)))).round().clamp_(min_int, max_int).reshape_as(x).transpose(0, axis)
    if n_bit < 8:
        x_int8 = pack_low_bit_tensor(x_int8, n_bit)
    return x_int8, scales, zeros


class KVCache(ABC, nn.Module):
    relevant_kwargs = ['max_cache_length', 'global_tokens', 'max_seq_length', 'cache_bits']

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, head_specific=False, variable_length=False, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.cache_shape = max_batch_size, n_heads, self.max_cache_length, head_dim
        self.quantize = self.cache_bits is not None
        self.n_bit = self.cache_bits
        self.quantization_axis = 2
        k_cache = torch.zeros(self.cache_shape, dtype=dtype)
        v_cache = torch.zeros(self.cache_shape, dtype=dtype)
        if self.quantize:
            k_cache, k_scales, k_zeros = quantize_tensor(k_cache, n_bit=self.n_bit, axis=self.quantization_axis)
            v_cache, v_scales, v_zeros = quantize_tensor(v_cache, n_bit=self.n_bit, axis=self.quantization_axis)
            self.register_buffer('k_scales', k_scales)
            self.register_buffer('v_scales', v_scales)
            self.register_buffer('k_zero_points', k_zeros)
            self.register_buffer('v_zero_points', v_zeros)
        self.register_buffer('k_cache', k_cache)
        self.register_buffer('v_cache', v_cache)
        self.n_heads = n_heads
        self.head_specific = head_specific
        self.register_buffer('pos', torch.full((max_batch_size, n_heads if head_specific else 1, self.max_cache_length), -1, dtype=torch.int))
        self.register_buffer('cache_cts', torch.zeros(n_heads if variable_length else 1, dtype=torch.int))
        kv_mask_shape = max_batch_size, n_heads, 1, self.max_cache_length
        self.register_buffer('mask', torch.zeros(kv_mask_shape, dtype=torch.bool))

    def reset(self):
        """
        Resets the cache to its initial state for a new example.

        NB: For more performance, don't reset k_cache and v_cache since we overwrite them in update.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.mask.zero_()
        self.cache_cts.zero_()
        self.pos.fill_(-1)

    def return_attn(self):
        """
        Returns whether the cache requires attention weights for cache management.
        """
        return False

    def memory_usage(self):
        tensors = []
        for obj in vars(self).values():
            if torch.is_tensor(obj):
                tensors.append(obj)
            elif isinstance(obj, dict):
                for vv in obj.values():
                    if torch.is_tensor(vv):
                        tensors.append(vv)
        return sum([(t.element_size() * t.numel()) for t in tensors]) / 1024 ** 3

    def compute_statistics(self, seq_len):
        """
        Computes statistics about the cache.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The cache size, the number of tokens inserted, and the compression ratio.
        """
        return {'compression_ratio': self.compression_ratio(seq_len).item(), 'cache_memory_gb': self.memory_usage()}

    def compression_ratio(self, seq_len):
        """
        Returns the compression ratio of the cache.
        """
        n = seq_len - 1
        assert torch.all(self.cache_cts <= self.max_cache_length)
        cache_size = self.cache_cts.clone().float()
        if self.n_bit is not None:
            cache_size *= self.n_bit / 16.0
        return ((n - cache_size) / n).mean()

    def quantize_cache(self):
        if self.quantize:
            self.k_cache, self.k_scales, self.k_zero_points = quantize_tensor(self.k_cache, n_bit=self.n_bit, axis=self.quantization_axis)
            self.v_cache, self.v_scales, self.v_zero_points = quantize_tensor(self.v_cache, n_bit=self.n_bit, axis=self.quantization_axis)

    def dequantize_cache(self):
        if self.quantize:
            self.k_cache = dequantize_tensor(self.k_cache, self.k_scales, self.k_zero_points, self.cache_shape, n_bit=self.n_bit, axis=self.quantization_axis)
            self.v_cache = dequantize_tensor(self.v_cache, self.v_scales, self.v_zero_points, self.cache_shape, n_bit=self.n_bit, axis=self.quantization_axis)

    def return_kv_cache(self):
        return self.k_cache, self.v_cache, self.mask

    def update_kv(self, input_pos, k_val, v_val, is_prefill, **kwargs):
        """
        Cache update logic.
        Takes in the input positions and the corresponding k and v values.
        Modifies self.pos, self.k_cache, self.v_cache place.

        Returns a tensor indicating the number of tokens inserted - number of tokens evicted.
        None is equivalent to 0.
        """
        self.dequantize_cache()
        if is_prefill:
            num_insertions = self._prefill_update(input_pos, k_val, v_val, **kwargs)
        else:
            num_insertions = self._decoding_update(input_pos, k_val, v_val, **kwargs)
        self.cache_cts += num_insertions[:len(self.cache_cts)]
        k, v, mask = self.return_kv_cache()
        self.quantize_cache()
        return k, v, mask

    def update_state(self, *args, **kwargs):
        """
        Optional method to update cache-specific internal state (excludes self.k_cache, self.v_cache, and self.pos).
        """
        pass

    def _decoding_update(self, input_pos, k_val, v_val, **kwargs):
        """
        Decoding logic for the cache.
        """
        eviction_idx = self._eviction_idx(input_pos)
        num_insertions = (self.pos.gather(2, eviction_idx.view(1, -1, 1)).squeeze() == -1).int().view(-1)
        self._fill(input_pos, k_val, v_val, fill_idxs=eviction_idx)
        return num_insertions

    def _eviction_idx(self, input_pos):
        scores = self._token_importances(input_pos)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        scores[:, :self.global_tokens] = float('inf')
        scores.masked_fill_(self.pos.view(scores.shape) == -1, float('-inf'))
        return torch.argmin(scores, dim=-1)

    def _prefill_update(self, input_pos, k_val, v_val, **kwargs):
        input_pos = input_pos.int()
        fill_idxs = torch.arange(input_pos.shape[-1], device=input_pos.device)
        self._fill_contiguous(input_pos, k_val, v_val, fill_idxs=fill_idxs)
        return torch.tensor([input_pos.shape[-1]], dtype=torch.int, device=input_pos.device)

    def _fill_contiguous(self, input_pos, k_val, v_val, fill_idxs: 'torch.Tensor | int', **kwargs):
        """
        A simple utility to fill the cache and pos.
        """
        self.pos[:, :, fill_idxs] = input_pos
        self.k_cache[:, :, fill_idxs, :] = k_val
        self.v_cache[:, :, fill_idxs, :] = v_val
        update_mask = kwargs.get('update_mask', True)
        if update_mask:
            self.mask[:, :, :, fill_idxs] = True

    @abstractmethod
    def _fill(self, input_pos, k_val, v_val, fill_idxs: 'torch.Tensor | int', **kwargs):
        """
        Modifies the cache in-place with key-value pairs at given fill_indices.

        Args:
            fill_indices (torch.Tensor): The indices specifying the positions to fill in the cache.
            input_pos (torch.Tensor): The input positions corresponding to the fill_indices.
            k_val (torch.Tensor): The key values to fill in the fill_indices slots.
            v_val (torch.Tensor): The value values to fill in the fill_indices slots.

        Returns:
            None
        """
        raise NotImplementedError

    def update_attn_history(self, attn):
        """
        Update the attention history with the most recent attention weights.
        """
        raise Exception(f'{self.__class__.__name__} requested return_attn=True but has not yet implemented a update_attn_history function.')


class KVCacheHeadConstant(KVCache):

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs)

    def _fill(self, input_pos, k_val, v_val, fill_idxs: 'torch.Tensor | int', **kwargs):
        return self._fill_contiguous(input_pos, k_val, v_val, fill_idxs, **kwargs)


class KVCacheHeadSpecific(KVCache):

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, variable_length=False, **kwargs):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, head_specific=True, variable_length=variable_length, **kwargs)

    def _fill(self, input_pos, k_val, v_val, fill_idxs: 'torch.Tensor | int', **kwargs):
        """
        Modifies the cache in-place with key-value pairs at given fill_indices.

        Args:
            fill_indices (torch.Tensor): The indices specifying the positions to fill in the cache.
            input_pos (torch.Tensor): The input positions corresponding to the fill_indices.
            k_val (torch.Tensor): The key values to fill in the fill_indices slots.
            v_val (torch.Tensor): The value values to fill in the fill_indices slots.

        Returns:
            None
        """
        assert input_pos.shape[-1] == k_val.shape[2] == v_val.shape[2]
        pos_fill_indices = fill_idxs.view(1, -1, 1)
        cache_fill_indices = fill_idxs.view(1, len(fill_idxs), 1, 1).expand(1, k_val.shape[1], 1, k_val.shape[-1])
        input_pos = input_pos.view(1, -1, 1).expand(1, k_val.shape[1], 1).int()
        self.pos.scatter_(2, pos_fill_indices, input_pos.int())
        self.k_cache.scatter_(2, cache_fill_indices, k_val)
        self.v_cache.scatter_(2, cache_fill_indices, v_val)
        update_mask = kwargs.get('update_mask', True)
        if update_mask:
            self.mask.scatter_(3, fill_idxs.view(1, -1, 1, 1), True)


class KVCacheFull(KVCacheHeadConstant):

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs):
        self.global_tokens = 0
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _eviction_idx(self, input_pos):
        return self.pos[0, 0].argmin().view(1)


class KVCacheRandom(KVCacheHeadConstant):
    relevant_kwargs = ['max_cache_length', 'max_seq_length', 'cache_bits', 'global_tokens', 'recent_window']

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _token_importances(self, input_pos):
        scores = torch.rand(self.max_cache_length, device=input_pos.device)
        scores[self.pos[0, 0] >= input_pos - self.recent_window] = float('inf')
        return scores


class KVCacheRecentGlobal(KVCacheHeadConstant):
    relevant_kwargs = ['max_cache_length', 'max_seq_length', 'cache_bits', 'global_tokens']

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _eviction_idx(self, input_pos):
        return (torch.argmin(self.pos[:, :, self.global_tokens:], dim=-1) + self.global_tokens).view(1)


class KVCacheL2(KVCacheHeadSpecific):
    relevant_kwargs = ['max_cache_length', 'max_seq_length', 'cache_bits', 'global_tokens', 'recent_window']

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)
        key_norm_shape = max_batch_size, n_heads, self.max_cache_length
        self.register_buffer('key_norm', torch.zeros(key_norm_shape, dtype=dtype))

    def reset(self):
        super().reset()
        self.key_norm.zero_()

    def _decoding_update(self, input_pos, k_val, v_val, **kwargs):
        fill_indices = self._eviction_idx(input_pos)
        num_insertions = (self.pos.gather(2, fill_indices.view(1, -1, 1)).squeeze() == -1).int().view(-1)
        self._fill(input_pos, k_val, v_val, fill_idxs=fill_indices)
        key_norm = torch.linalg.vector_norm(k_val, ord=2, dim=-1)
        self.key_norm.scatter_(2, fill_indices.view(1, -1, 1), key_norm)
        return num_insertions

    def _token_importances(self, input_pos):
        return (self.key_norm.max() - self.key_norm).masked_fill(self.pos >= input_pos - self.recent_window, float('inf')).squeeze(0)

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        pass
        if is_prefill:
            self.key_norm.copy_(torch.linalg.vector_norm(self.k_cache, ord=2, dim=-1))


class KVCacheHeavyHitter(KVCacheHeadSpecific):
    relevant_kwargs = ['max_cache_length', 'max_seq_length', 'cache_bits', 'global_tokens', 'history_window_size', 'recent_window', 'attn_thresholding']

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, variable_length=False, **kwargs):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, variable_length, **kwargs)
        history_num_shape = max_batch_size, n_heads, self.max_cache_length, self.history_window_size
        history_denom_shape = max_batch_size, n_heads, self.max_cache_length
        history_num_dtype = torch.bool if self.attn_thresholding else torch.float64 if self.history_window_size == 1 else dtype
        self.register_buffer('attn_history_num', torch.zeros(history_num_shape, dtype=history_num_dtype))
        self.register_buffer('attn_history_denom', torch.zeros(history_denom_shape, dtype=torch.int32))
        self.register_buffer('attn_counter', torch.zeros((1,), dtype=torch.int64))

    def reset(self):
        super().reset()
        self.attn_history_num.zero_()
        self.attn_history_denom.zero_()
        self.attn_counter.zero_()

    def return_attn(self) ->bool:
        return True

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        """
        Insert the most recent attention into the history buffer.

        If self.attn_thresholding = True, insert a binary indicator of whether the attention >= uniform attention.
        """
        seq_len = attn.shape[-1]
        if is_prefill and attn.ndim == 4:
            attn = attn.squeeze(0).sum(dim=1) / (seq_len - input_pos)
        attn = attn.view(1, self.n_heads, -1, 1)
        attn = (attn >= 1 / self.cache_cts).int() if self.attn_thresholding else attn
        padding = max(self.max_cache_length - seq_len, 0)
        pad_attn = torch.zeros(1, self.n_heads, padding, 1, dtype=attn.dtype, device=attn.device)
        attn = torch.cat([attn, pad_attn], dim=2)
        history_idx = self.attn_counter % self.history_window_size
        if self.history_window_size == 1:
            self.attn_history_num[:, :, :, history_idx] += attn
        else:
            self.attn_history_num[:, :, :, history_idx] = attn
        self.attn_history_denom += 1
        self.attn_counter += 1

    def _eviction_idx(self, input_pos):
        numerator = self.attn_history_num.sum(dim=-1).float()
        if self.history_window_size == 1:
            denominator = self.attn_history_denom.clamp_min(1)
        else:
            denominator = self.attn_history_denom.clamp(1, self.history_window_size)
        avg_attn = numerator / denominator
        avg_attn.masked_fill_(torch.logical_or(self.pos < self.global_tokens, self.pos >= input_pos - self.recent_window), 1.0)
        avg_attn.masked_fill_(self.pos == -1, 0.0)
        fill_idxs = avg_attn.argmin(dim=-1).squeeze()
        num_fill = fill_idxs.view(1, -1, 1, 1).expand(1, -1, 1, self.attn_history_num.shape[-1])
        denom_fill = fill_idxs.view(1, -1, 1)
        self.attn_history_num.scatter_(2, num_fill, torch.zeros_like(num_fill, dtype=self.attn_history_num.dtype))
        self.attn_history_denom.scatter_(2, denom_fill, torch.zeros_like(denom_fill, dtype=torch.int32))
        return fill_idxs


def create_window_attention_mask(seq_len, window_size, device, global_tokens: 'int'=4):
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    mask[:, :global_tokens] = True
    for i in range(seq_len):
        mask[i, max(0, i + 1 - window_size):i + 1] = True
    return mask


class KVCacheHybrid(KVCacheHeavyHitter):
    relevant_kwargs = ['max_cache_length', 'max_seq_length', 'cache_bits', 'global_tokens', 'token_ids', 'min_recovery_frac', 'hybrid_strategies']

    def __init__(self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs):
        self.attn_thresholding = False
        self.history_window_size = 400
        self.recent_window = None
        super().__init__(max_batch_size, n_heads, head_dim, dtype, variable_length=True, **kwargs)
        self.requires_special = any([('special' in strat['strategy']) for strat in self.hybrid_strategies])
        mask_shape = max_batch_size, n_heads, self.max_cache_length
        if self.requires_special:
            special_ids = [torch.tensor(ids) for ids in kwargs['token_ids']['special']]
            self.register_buffer('special_ids', torch.nested.nested_tensor(special_ids))
            self.register_buffer('special_mask', torch.zeros(mask_shape, dtype=torch.bool))
            self.register_buffer('num_special', torch.zeros((1,), dtype=torch.int))
        self.requires_punc = any([('punc' in strat['strategy']) for strat in self.hybrid_strategies])
        if self.requires_punc:
            punc_ids = torch.Tensor(kwargs['token_ids']['punctuation'])
            self.register_buffer('punc_ids', punc_ids)
            self.register_buffer('punc_mask', torch.zeros(mask_shape, dtype=torch.bool))
            self.register_buffer('num_punc', torch.zeros((1,), dtype=torch.int))
        self.requires_heavy_hitter = self._init_requires_heavy_hitter()
        kv_mask_shape = max_batch_size, n_heads, 1, self.max_cache_length
        self.register_buffer('mask', torch.zeros(kv_mask_shape, dtype=torch.bool))

    def return_attn(self):
        return self.requires_heavy_hitter

    def _init_requires_heavy_hitter(self):
        return any([('heavy_hitter' in strat['strategy']) for strat in self.hybrid_strategies])

    def _eviction_idx_for_head(self, head_idx, input_pos, recent_window, apply_heavy_hitter=False, apply_window=False, apply_special=False, apply_punc=False):
        if apply_heavy_hitter:
            numerator = self.attn_history_num[:, head_idx, :self.cache_cts[head_idx]].sum(dim=-1).float()
            if self.history_window_size == 1:
                denominator = self.attn_history_denom[:, head_idx, :self.cache_cts[head_idx]]
            else:
                denominator = self.attn_history_denom[:, head_idx, :self.cache_cts[head_idx]].clamp_max(self.history_window_size)
            score = numerator / denominator
        else:
            score = self.pos[:, head_idx, :self.cache_cts[head_idx]].clone().float()
        save_mask = torch.zeros_like(score, dtype=torch.bool)
        save_mask[:, :self.global_tokens] = 1
        if apply_special:
            save_mask |= self.special_mask[:, head_idx, :self.cache_cts[head_idx]]
        if apply_punc:
            save_mask |= self.punc_mask[:, head_idx, :self.cache_cts[head_idx]]
        if apply_window:
            window_mask = self.pos[:, head_idx, :self.cache_cts[head_idx]] > input_pos - recent_window
            save_mask |= window_mask
        score.masked_fill_(save_mask, float('inf'))
        fill_idx = score.argmin(dim=-1)
        return fill_idx

    def _select_fill_idx(self, strategy, head_idx, input_pos, is_punc: 'bool'=False):

        def _end_idx():
            return min(self.max_cache_length - 1, self.cache_cts[head_idx].clone())
        strategy = self.hybrid_strategies[strategy]
        name = strategy['strategy']
        if 'punc' in name and is_punc:
            return _end_idx(), False
        if name == 'full':
            return _end_idx(), False
        budget = torch.tensor([self.global_tokens], dtype=torch.int, device=input_pos.device)
        if 'special' in name:
            budget += self.num_special
        if 'punc' in name:
            budget += self.num_punc
        if 'window' in name:
            budget += round(strategy['recent_window'] * self.max_cache_length)
        if 'heavy_hitter' in name:
            budget += round(strategy['heavy_hitter_frac'] * self.max_cache_length)
        eviction_required = self.cache_cts[head_idx] >= budget
        if not eviction_required:
            return _end_idx(), False
        if 'heavy_hitter' in name or 'window' in name:
            recent_window = round(strategy.get('recent_window', 0) * self.max_cache_length)
            fill_idx = self._eviction_idx_for_head(head_idx, input_pos, recent_window=recent_window, apply_heavy_hitter='heavy_hitter' in name, apply_window='window' in name, apply_punc='punc' in name, apply_special='special' in name)
            return fill_idx, True
        assert 'punc' in name or 'special' in name, f'Invalid hybrid strategy {name}'
        return None, False

    def reset(self):
        super().reset()
        self.cache_strategies.fill = None
        self.requires_heavy_hitter = self._init_requires_heavy_hitter()
        if hasattr(self, 'special_mask'):
            self.special_mask.zero_()
            self.num_special.zero_()
        if hasattr(self, 'punc_mask'):
            self.punc_mask.zero_()
            self.num_punc.zero_()

    def _decoding_update(self, input_pos, k_val, v_val, **kwargs):
        input_ids = kwargs.get('input_ids')
        n_heads = k_val.shape[1]
        is_punc = torch.isin(input_ids, self.punc_ids) if hasattr(self, 'punc_ids') else False
        fill_indices = torch.full((n_heads,), self.max_cache_length - 1, dtype=torch.int64, device=k_val.device)
        cache_ct_incr = torch.zeros_like(fill_indices)
        for head_idx, strategy in enumerate(self.cache_strategies):
            fill_idx, eviction_required = self._select_fill_idx(strategy, head_idx, input_pos, is_punc=is_punc)
            if fill_idx is None:
                continue
            fill_indices[head_idx] = fill_idx
            if eviction_required:
                if self.requires_heavy_hitter:
                    self.attn_history_num[:, head_idx, fill_idx, :].fill_(0)
                    self.attn_history_denom[:, head_idx, fill_idx].fill_(0)
            else:
                cache_ct_incr[head_idx] = 1
                self.mask[:, head_idx, :, fill_idx] = True
        kwargs = {'update_mask': False}
        self._fill(input_pos, k_val, v_val, fill_indices, **kwargs)
        if is_punc and hasattr(self, 'num_punc'):
            self.punc_mask.scatter_(2, fill_indices.view(1, -1, 1), is_punc.view(1, 1, 1).expand(1, n_heads, 1))
            self.num_punc += 1
        return cache_ct_incr

    def build_special_ids_mask(self, input_ids):
        seq_len = input_ids.shape[-1]
        special_ids_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in self.special_ids:
            id_len = len(special_id)
            if id_len == 1:
                special_ids_mask[torch.where(input_ids == special_id)[0]] = True
            else:
                for i in range(seq_len - id_len + 1):
                    if torch.equal(input_ids[i:i + id_len], special_id):
                        special_ids_mask[i:i + id_len] = True
        return special_ids_mask

    def build_punc_ids_mask(self, input_ids):
        if self.punc_ids.device != input_ids.device:
            self.punc_ids = self.punc_ids
        punc_ids_mask = torch.isin(input_ids, self.punc_ids)
        return punc_ids_mask

    def compute_statistics(self, seq_len):
        stats = super().compute_statistics(seq_len)
        cts = Counter([self.hybrid_strategies[i]['strategy'] for i in self.cache_strategies.tolist()])
        stats['avg_strategy_idx'] = sum(self.cache_strategies.tolist()) / len(self.cache_strategies)
        stats.update({strategy: (cts.get(strategy, 0) / len(self.cache_strategies)) for strategy in sorted(list(set([x['strategy'] for x in self.hybrid_strategies])))})
        return stats

    def build_masks(self, cum_attn, special_mask, punc_mask, total_len):
        device = cum_attn.device
        n_heads, seq_len = cum_attn.shape
        masks = []
        for s in self.hybrid_strategies:
            strat_mask = torch.zeros(n_heads, seq_len, seq_len, dtype=torch.bool, device=device)
            strat_mask[:, :, :self.global_tokens] = True
            name = s['strategy']
            if 'special' in name:
                strat_mask |= special_mask.view(1, 1, -1).expand(n_heads, seq_len, seq_len)
            if 'punc' in name:
                strat_mask |= punc_mask.view(1, 1, -1).expand(n_heads, seq_len, seq_len)
            if 'window' in name:
                assert 'recent_window' in s and s['recent_window'] <= 1, 'Window strategy should have recent_window expressed as a fraction <= 1.'
                strat_mask |= create_window_attention_mask(seq_len, max(1, int(s['recent_window'] * total_len)), device, global_tokens=self.global_tokens).unsqueeze(0).expand(n_heads, seq_len, seq_len)
            if 'heavy_hitter' in name:
                avail_idxs = torch.where(~strat_mask[0, -1, :])[0]
                attn_slice = cum_attn.gather(1, avail_idxs.unsqueeze(0).expand(n_heads, -1))
                num_hh = math.ceil(min(s['heavy_hitter_frac'] * total_len, len(avail_idxs)))
                heavy_hitters = attn_slice.topk(num_hh, dim=1, largest=True).indices.sort(dim=1).values
                heavy_hitters_idx = avail_idxs.view(1, -1).expand(n_heads, -1).gather(1, heavy_hitters)
                strat_mask.scatter_(2, heavy_hitters_idx.view(n_heads, 1, num_hh).expand(n_heads, seq_len, num_hh), True)
            if name == 'full':
                strat_mask.fill_(True)
            masks.append(strat_mask)
        return torch.stack(masks)

    def profile_attn_heads(self, input_pos, attn, **kwargs):
        input_ids = kwargs['input_ids']
        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[-1]
        special_mask = punc_mask = None
        if self.requires_special:
            special_mask = self.build_special_ids_mask(input_ids)
            self.num_special = special_mask.sum()
        if self.requires_punc:
            punc_mask = self.build_punc_ids_mask(input_ids)
            self.num_punc = punc_mask.sum()
        cum_attn = None
        if any([('heavy_hitter' in s['strategy']) for s in self.hybrid_strategies]):
            cum_attn = attn.squeeze(0).sum(dim=1) / (seq_len - input_pos)
        masks_for_scoring = self.build_masks(cum_attn, special_mask, punc_mask, total_len=seq_len)
        attn_rep = attn.expand(masks_for_scoring.shape[0], -1, -1, -1)
        compressed_scores = attn_rep.masked_fill(~masks_for_scoring, 0).sum(dim=-1).mean(dim=-1)
        cache_strategies = (compressed_scores >= self.min_recovery_frac).int().argmax(dim=0)
        assert self.max_cache_length >= seq_len
        masks_for_filling = self.build_masks(cum_attn, special_mask, punc_mask, total_len=self.max_cache_length)
        masks_all = masks_for_filling[:, :, -1, :].transpose(1, 0)
        mask_optimal = masks_all.gather(1, cache_strategies.view(-1, 1, 1).expand(-1, -1, seq_len)).squeeze(1)
        return cache_strategies, special_mask, punc_mask, mask_optimal, cum_attn

    def profile_and_update(self, input_pos, k_val, v_val, attn, **kwargs):
        """
        Profile the attention heads to determine the optimal KV-cache allocation.
        """
        input_ids = kwargs['input_ids']
        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[-1]
        n_heads = attn.shape[1]
        dim = k_val.shape[-1]
        self.cache_strategies, special_mask, punc_mask, mask_optimal, cum_attn = self.profile_attn_heads(input_pos, attn, **kwargs)
        self.requires_heavy_hitter = any([('heavy_hitter' in self.hybrid_strategies[i]['strategy']) for i in self.cache_strategies])
        self.requires_punc = any([('punc' in self.hybrid_strategies[i]['strategy']) for i in self.cache_strategies])
        self.requires_special = any([('special' in self.hybrid_strategies[i]['strategy']) for i in self.cache_strategies])
        order = mask_optimal.int().argsort(dim=1, descending=True)
        order_exp = order.view(1, n_heads, seq_len, 1).expand(-1, -1, -1, dim)
        k_val = k_val.gather(2, order_exp)
        v_val = v_val.gather(2, order_exp)
        input_pos = input_pos.unsqueeze(0).expand(n_heads, -1).gather(1, order).int()
        fill_idxs = torch.arange(seq_len, device=input_pos.device)
        self._fill_contiguous(input_pos, k_val, v_val, fill_idxs)
        self.cache_cts = mask_optimal.sum(dim=1)
        for head_idx in range(n_heads):
            self.pos[:, head_idx, self.cache_cts[head_idx]:].fill_(-1)
            self.k_cache[:, head_idx, self.cache_cts[head_idx]:].fill_(0)
            self.v_cache[:, head_idx, self.cache_cts[head_idx]:].fill_(0)
        if hasattr(self, 'special_mask'):
            special_mask = special_mask.view(1, -1).expand(n_heads, -1).gather(1, order)
            self.special_mask[:, :, :seq_len] = special_mask
        if hasattr(self, 'punc_mask'):
            punc_mask = punc_mask.view(1, -1).expand(n_heads, -1).gather(1, order)
            self.punc_mask[:, :, :seq_len] = punc_mask
        range_mask = torch.arange(seq_len, device=self.mask.device).view(1, -1).expand(n_heads, -1)
        self.mask[:, :, :, :seq_len] = (range_mask < self.cache_cts.view(-1, 1).expand(-1, seq_len)).view(-1, n_heads, 1, seq_len)
        if self.requires_heavy_hitter:
            cum_attn = cum_attn.gather(1, order).unsqueeze(0)
            super().update_state(input_pos, k_val, v_val, is_prefill=True, attn=cum_attn, **kwargs)

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        """
        Insert the most recent attention into the history buffer.

        If self.attn_thresholding = True, insert a binary indicator of whether the attention >= uniform attention.
        """
        if is_prefill:
            self.profile_and_update(input_pos, k_val, v_val, attn, **kwargs)
        elif self.requires_heavy_hitter:
            super().update_state(input_pos, k_val, v_val, is_prefill, attn, **kwargs)
        else:
            assert attn is None, 'Attn should be None if no attention is required.'

