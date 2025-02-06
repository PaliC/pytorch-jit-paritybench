import sys
_module = sys.modules[__name__]
del sys
benchmark_generation = _module
benchmark_training_throughput = _module
benchmark_cross_entropy = _module
benchmark_layernorm = _module
benchmark = _module
benchmark_abc = _module
benchmark_based = _module
benchmark_delta_rule = _module
benchmark_fla = _module
benchmark_gla = _module
benchmark_gsa = _module
benchmark_hgrn = _module
benchmark_retention = _module
benchmark_rwkv6 = _module
benchmark_simple_gla_vs_mamba2 = _module
harness = _module
ppl = _module
fla = _module
layers = _module
abc = _module
attn = _module
based = _module
bitattn = _module
delta_net = _module
gated_deltanet = _module
gla = _module
gsa = _module
hgrn = _module
hgrn2 = _module
lightnet = _module
linear_attn = _module
multiscale_retention = _module
rebased = _module
rwkv6 = _module
rwkv7 = _module
simple_gla = _module
models = _module
configuration_abc = _module
modeling_abc = _module
bitnet = _module
configuration_bitnet = _module
modeling_bitnet = _module
configuration_delta_net = _module
modeling_delta_net = _module
configuration_gated_deltanet = _module
modeling_gated_deltanet = _module
configuration_gla = _module
modeling_gla = _module
configuration_gsa = _module
modeling_gsa = _module
configuration_hgrn = _module
modeling_hgrn = _module
configuration_hgrn2 = _module
modeling_hgrn2 = _module
configuration_lightnet = _module
modeling_lightnet = _module
configuration_linear_attn = _module
modeling_linear_attn = _module
mamba = _module
configuration_mamba = _module
modeling_mamba = _module
mamba2 = _module
configuration_mamba2 = _module
modeling_mamba2 = _module
retnet = _module
configuration_retnet = _module
modeling_retnet = _module
configuration_rwkv6 = _module
modeling_rwkv6 = _module
configuration_rwkv7 = _module
modeling_rwkv7 = _module
samba = _module
configuration_samba = _module
modeling_samba = _module
transformer = _module
configuration_transformer = _module
modeling_transformer = _module
utils = _module
modules = _module
activations = _module
convolution = _module
feature_map = _module
fused_bitlinear = _module
fused_cross_entropy = _module
fused_kl_div = _module
fused_linear_cross_entropy = _module
fused_norm_gate = _module
l2norm = _module
layernorm = _module
layernorm_gated = _module
rotary = _module
ops = _module
chunk = _module
naive = _module
fused_chunk = _module
naive = _module
parallel = _module
common = _module
chunk_delta_h = _module
chunk_h = _module
chunk_h_parallel = _module
chunk_h_split = _module
chunk_o = _module
fused_recurrent = _module
utils = _module
delta_rule = _module
chunk = _module
fused_recurrent = _module
naive = _module
parallel = _module
wy_fast = _module
gated_delta_rule = _module
chunk = _module
fused_recurrent = _module
wy_fast = _module
generalized_delta_rule = _module
dplr = _module
chunk = _module
chunk_A_bwd = _module
chunk_A_fwd = _module
chunk_h_bwd = _module
chunk_h_fwd = _module
chunk_o_bwd = _module
chunk_o_fwd = _module
fused_recurrent = _module
naive = _module
wy_fast_bwd = _module
wy_fast_fwd = _module
iplr = _module
chunk = _module
fused_recurrent = _module
naive = _module
wy_fast = _module
chunk = _module
fused_chunk = _module
fused_recurrent = _module
naive = _module
chunk = _module
fused_recurrent = _module
naive = _module
chunk = _module
fused_recurrent = _module
naive = _module
chunk = _module
fused_chunk = _module
fused_recurrent = _module
naive = _module
utils = _module
naive = _module
parallel = _module
retention = _module
chunk = _module
fused_chunk = _module
fused_recurrent = _module
naive = _module
parallel = _module
rwkv4 = _module
fused_recurrent = _module
chunk = _module
chunk_naive = _module
fused_recurrent = _module
recurrent_naive = _module
chunk = _module
fused_recurrent = _module
chunk = _module
fused_recurrent = _module
naive = _module
parallel = _module
ttt = _module
chunk = _module
fused_chunk = _module
naive = _module
cumsum = _module
exp = _module
logcumsumexp = _module
logsumexp = _module
matmul = _module
softmax = _module
utils = _module
flame = _module
data = _module
logging = _module
parser = _module
preprocess = _module
run = _module
setup = _module
test_based = _module
test_gla = _module
test_conv = _module
test_cross_entropy = _module
test_kl_div = _module
test_layernorm = _module
test_rotary = _module
test_based = _module
test_cumsum = _module
test_delta = _module
test_dplr_delta = _module
test_gated_delta = _module
test_gla = _module
test_gsa = _module
test_hgrn = _module
test_iplr_delta = _module
test_linear_attn = _module
test_retention = _module
test_rwkv6 = _module
test_simple_gla = _module
test_ttt = _module
test_utils = _module
test_fused_chunk = _module
test_padding = _module
test_rwkv7_conversion = _module
convert_from_llama = _module
convert_from_rwkv6 = _module
convert_from_rwkv7 = _module

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


import time


import torch


from typing import Optional


from typing import Tuple


from torch.cuda import max_memory_allocated


from torch.cuda import memory_allocated


from torch.optim import AdamW


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.benchmark as benchmark


from torch.nn import functional as F


import math


from functools import partial


from typing import Any


from typing import Dict


from typing import List


from typing import Iterator


from typing import Union


from torch.nn.utils.rnn import pad_sequence


import warnings


from typing import TYPE_CHECKING


import torch.utils.checkpoint


from torch import nn


from typing import cast


from torch import Tensor


from torch.autograd.function import Function


from torch.autograd.function import FunctionCtx


from torch.autograd.function import once_differentiable


import functools


from typing import Callable


from copy import deepcopy


from typing import Iterable


import numpy as np


from itertools import chain


import re


def contiguous(fn: 'Callable[..., torch.Tensor]') ->Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous.
    """

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx, *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args), **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def layer_norm_bwd(dy: 'torch.Tensor', x: 'torch.Tensor', weight: 'torch.Tensor', bias: 'torch.Tensor', eps: 'float', mean: 'torch.Tensor', rstd: 'torch.Tensor', z: 'torch.Tensor'=None, group_size: 'int'=None, norm_before_gate: 'bool'=True, is_rms_norm: 'bool'=False, recompute_output: 'bool'=False, dz: 'torch.Tensor'=None, out: 'torch.Tensor'=None):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    dx = torch.empty_like(x)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
        assert dz.stride(-1) == 1
    else:
        dz = torch.empty_like(z) if z is not None else None
    if recompute_output:
        if out is None:
            out = torch.empty_like(x)
        assert out.shape == x.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nrow_groups = math.ceil(sm_count * math.ceil(4 / num_warps) / ngroups)
    _dw = torch.empty((nrow_groups, N), dtype=torch.float32, device=weight.device)
    _db = torch.empty((nrow_groups, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / nrow_groups)
    grid = nrow_groups, ngroups
    with torch.device(x.device.index):
        layer_norm_bwd_kernel[grid](x, weight, bias, z, out if recompute_output else None, dy, dx, _dw, _db, dz, mean, rstd, x.stride(0), z.stride(0) if z is not None else 0, 0 if not recompute_output else out.stride(0), dy.stride(0), dx.stride(0), dz.stride(0) if dz is not None else 0, _dw.stride(0), _db.stride(0) if _db is not None else 0, M, group_size, eps, rows_per_program, BLOCK_N=BLOCK_N, NORM_BEFORE_GATE=norm_before_gate, IS_RMS_NORM=is_rms_norm, num_warps=num_warps)
    dw = _dw.sum(0)
    db = _db.sum(0) if bias is not None else None
    return (dx, dw, db, dz) if not recompute_output else (dx, dw, db, dz, out)


def layer_norm_fwd(x: 'torch.Tensor', weight: 'torch.Tensor', bias: 'torch.Tensor', eps: 'float', z: 'torch.Tensor'=None, out: 'torch.Tensor'=None, group_size: 'int'=None, norm_before_gate: 'bool'=True, is_rms_norm: 'bool'=False):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = M, ngroups
    with torch.device(x.device.index):
        layer_norm_fwd_kernel[grid](x, out, weight, bias, z, mean, rstd, x.stride(0), out.stride(0), z.stride(0) if z is not None else 0, M, group_size, eps, BLOCK_N=BLOCK_N, NORM_BEFORE_GATE=norm_before_gate, IS_RMS_NORM=is_rms_norm, num_warps=num_warps)
    return out, mean, rstd


class LayerNormSwishGateFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, o, weight, bias, residual=None, eps=1e-06, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
        x_shape_og = x.shape
        o_shape_og = o.shape
        x = x.reshape(-1, x.shape[-1])
        o = o.reshape(-1, o.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = residual.dtype if residual is not None else torch.float if residual_in_fp32 else None
        y, mean, rstd, residual_out = layer_norm_fwd(x, o, weight, bias, eps, residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm)
        ctx.save_for_backward(residual_out, o, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.o_shape_og = o_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dy, *args):
        x, o, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, do, dw, db, dresidual_in = layer_norm_bwd(dy, x, o, weight, bias, ctx.eps, mean, rstd, dresidual, ctx.has_residual, ctx.is_rms_norm, x_dtype=ctx.x_dtype)
        return dx.reshape(ctx.x_shape_og), do.reshape(ctx.o_shape_og), dw, db, dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None, None, None, None, None


def rms_norm_swish_gate_fn(x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-06):
    return LayerNormSwishGateFn.apply(x, o, weight, bias, residual, eps, prenorm, residual_in_fp32, True)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, weight, bias, residual=None, eps=1e-05, prenorm=False, residual_in_fp32=False, is_rms_norm=False, num_groups=1):
        x_shape_og = x.shape
        if x.shape[-1] % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        x = x.reshape(-1, x.shape[-1] // num_groups)
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape_as(x)
        residual_dtype = residual.dtype if residual is not None else torch.float32 if residual_in_fp32 else None
        y, mean, rstd, residual_out = layer_norm_fwd(x, weight, bias, eps, residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm, num_groups=num_groups)
        ctx.save_for_backward(residual_out, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.num_groups = num_groups
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dy, *args):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1] // ctx.num_groups)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, x.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in = layer_norm_bwd(dy, x, weight, bias, ctx.eps, mean, rstd, dresidual, ctx.has_residual, ctx.is_rms_norm, x_dtype=ctx.x_dtype, num_groups=ctx.num_groups)
        return dx.reshape(ctx.x_shape_og), dw, db, dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None, None, None, None, None, None


def rms_norm(x: 'torch.Tensor', weight: 'torch.Tensor', bias: 'torch.Tensor', residual: 'torch.Tensor'=None, eps: 'float'=1e-05, prenorm: 'bool'=False, residual_in_fp32: 'bool'=False):
    return LayerNormFunction.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, True)


@contiguous
def rotary_embedding_fwdbwd(x: 'torch.Tensor', cos: 'torch.Tensor', sin: 'torch.Tensor', seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None, interleaved: 'bool'=False, inplace: 'bool'=False, conjugate: 'bool'=False) ->torch.Tensor:
    """
    Args:
        x: (N, T, H, D).
        cos: (TR, R / 2)
        sin: (TR, R / 2)
        seqlen_offsets: integer or integer tensor of size (N,)
        cu_seqlens: (N + 1,) or None
        max_seqlen: int

    Returns:
        y: [N, T, H, D]
    """
    is_varlen = cu_seqlens is not None
    B, T, H, D = x.shape
    if not is_varlen:
        N = B
    else:
        assert max_seqlen is not None, 'If cu_seqlens is passed in, then max_seqlen must be passed'
        N, T = cu_seqlens.shape[0] - 1, max_seqlen
    TR, R = cos.shape
    assert sin.shape == cos.shape
    R2 = R * 2
    assert D <= 256, 'Only support D <= 256'
    assert TR >= T, 'TR must be >= T'
    assert cos.dtype == sin.dtype, f'cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}'
    assert x.dtype == cos.dtype, f'Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}'
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (N,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
    else:
        assert seqlen_offsets + T <= TR
    y = torch.empty_like(x) if not inplace else x
    if R2 < D and not inplace:
        y[..., R2:].copy_(x[..., R2:])
    BD = triton.next_power_of_2(R2)

    def grid(META):
        return triton.cdiv(T, META['BT']), N, H
    with torch.device(x.device.index):
        rotary_embedding_kernel[grid](x, cos, sin, y, cu_seqlens, seqlen_offsets, B=B, T=T, H=H, D=D, R=R, TR=TR, BD=BD, IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor), IS_VARLEN=is_varlen, INTERLEAVED=interleaved, CONJUGATE=conjugate)
    return y


class RotaryEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False, seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None):
        y = rotary_embedding_fwdbwd(x, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=interleaved, inplace=inplace)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return y if not inplace else x

    @staticmethod
    @contiguous
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = rotary_embedding_fwdbwd(do, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=ctx.max_seqlen, interleaved=ctx.interleaved, inplace=ctx.inplace, conjugate=True)
        return dx, None, None, None, None, None, None, None


def rotary_embedding(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None):
    """
    Args:
        x: [N, T, H, D]
        cos, sin: [TR, R//2]
        interleaved:
            If True, rotate pairs of even and odd dimensions (GPT-J style) instead of 1st half and 2nd half (GPT-NeoX style).
        inplace:
            If True, apply rotary embedding in-place.
        seqlen_offsets: [N,] or int.
            Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: [N + 1,] or None
        max_seqlen: int

    Returns:
        out: [N, T, H, D]
    """
    return RotaryEmbeddingFunction.apply(x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen)


class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: 'int', base: 'float'=10000.0, scale_base: 'Optional[float]'=None, interleaved: 'bool'=False, pos_idx_in_fp32: 'bool'=True, device: 'Optional[torch.device]'=None):
        """
        interleaved:
            If True, rotate pairs of even and odd dimensions (GPT-J style) instead of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32:
            If True, the position indices [0.0, ..., seqlen - 1] are in fp32, otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq.
            In most cases this would be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16, we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.scale_base = scale_base
        self.interleaved = interleaved
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.device = device
        self.register_buffer('inv_freq', torch.empty(-(dim // -2), dtype=torch.float32, device=device), persistent=False)
        scale = None
        if scale_base is not None:
            scale = torch.empty(-(dim // -2), dtype=torch.float32, device=device)
        self.register_buffer('scale', scale, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.inv_freq.copy_(self._compute_inv_freq(device=self.inv_freq.device))
            if self.scale_base is not None:
                self.scale.copy_(self._compute_scale(device=self.scale.device))

    def __repr__(self):
        s = f'{self.__class__.__name__}('
        s += f'dim={self.dim}, '
        s += f'base={self.base}, '
        s += f'interleaved={self.interleaved}, '
        if self.scale_base is not None:
            s += f'scale_base={self.scale_base}, '
        s += f'pos_idx_in_fp32={self.pos_idx_in_fp32})'
        return s

    def _compute_inv_freq(self, device=None):
        return 1.0 / self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)

    def _compute_scale(self, device=None):
        return (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) + 0.4 * self.dim) / (1.4 * self.dim)

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if seqlen > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device or self._cos_cached.dtype != dtype or self.training and self._cos_cached.is_inference():
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs)
                self._sin_cached = torch.sin(freqs)
            else:
                power = (torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2) / self.scale_base
                scale = self.scale ** rearrange(power, 's -> s 1')
                self._cos_cached = torch.cos(freqs) * scale
                self._sin_cached = torch.sin(freqs) * scale
                self._cos_k_cached = torch.cos(freqs) / scale
                self._sin_k_cached = torch.sin(freqs) / scale

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', seqlen_offset: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        q: [N, T, H, D]
        k: [N, T, H, D]
        seqlen_offset:
            (N,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (N,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        cu_seqlens: (N + 1,) or None
        max_seqlen: int
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype)
        if self.scale is None:
            q = rotary_embedding(q, self._cos_cached, self._sin_cached, interleaved=self.interleaved, seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            k = rotary_embedding(k, self._cos_cached, self._sin_cached, interleaved=self.interleaved, seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        else:
            q = rotary_embedding(q, self._cos_cached, self._sin_cached, interleaved=self.interleaved, seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            k = rotary_embedding(k, self._cos_k_cached, self._sin_k_cached, interleaved=self.interleaved, seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return q, k


@torch.jit.script
def bias_gelu(y, bias):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def bias_gelu_bwd(g, y, bias):
    """Assume that y has shape (B, D) and bias has shape (D)"""
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    grad_y = ff * g
    return grad_y, grad_y.sum(dim=0, dtype=bias.dtype)


class GeLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


@torch.jit.script
def gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


@torch.jit.script
def gelu_fwd(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


class FastGeLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, input)
        return tmp


fast_gelu_impl = FastGeLUFunction.apply


def logsigmoid_bwd(x: 'torch.Tensor', dy: 'torch.Tensor', temperature: 'float'=1.0) ->torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(T, torch.cuda.get_device_properties(x.device).multi_processor_count))
    dx = torch.empty_like(x)
    logsigmoid_bwd_kernel[triton.cdiv(T, B),](x=x, dx=dx, dy=dy, temperature=temperature, T=T, D=D, B=B)
    return dx


def logsigmoid_fwd(x: 'torch.Tensor', temperature: 'float'=1.0) ->torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(T, torch.cuda.get_device_properties(x.device).multi_processor_count))
    y = torch.empty_like(x)
    logsigmoid_fwd_kernel[triton.cdiv(T, B),](x=x, y=y, temperature=temperature, T=T, D=D, B=B)
    return y


class LogSigmoidFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, temperature):
        ctx.save_for_backward(x)
        ctx.temperature = temperature
        return logsigmoid_fwd(x, temperature)

    @staticmethod
    @contiguous
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return logsigmoid_bwd(x, dy, ctx.temperature), None


def logsigmoid(x: 'torch.Tensor', temperature: 'float'=1.0) ->torch.Tensor:
    return LogSigmoidFunction.apply(x, temperature)


sigmoid_bwd_codestring = """
template <typename T> T sigmoid_bwd(T x, T g) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(g) * x_sigmoid * (1.0f - x_sigmoid);
}
"""


sigmoid_bwd = torch.cuda.jiterator._create_jit_fn(sigmoid_bwd_codestring)


sigmoid_fwd_codestring = """
template <typename T> T sigmoid_fwd(T x) {
    return 1.0f / (1.0f + ::exp(-float(x)));
}
"""


sigmoid_fwd = torch.cuda.jiterator._create_jit_fn(sigmoid_fwd_codestring)


class SigmoidFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return sigmoid_bwd(x, dout)


sigmoid = SigmoidFunction.apply


@torch.jit.script
def sqrelu_bwd(g, x):
    return 2.0 * g * F.relu(x)


@torch.jit.script
def sqrelu_fwd(x):
    r = F.relu(x)
    return r * r


class SquaredReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return sqrelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return sqrelu_bwd(grad_output, input)


sqrelu = SquaredReLUFunction.apply


swish_bwd_codestring = """
template <typename T> T swish_bwd(T x, T g) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(g) * x_sigmoid * (1.0f - float(x) * x_sigmoid + float(x));
}
"""


swish_bwd = torch.cuda.jiterator._create_jit_fn(swish_bwd_codestring)


swish_fwd_codestring = """
template <typename T> T swish_fwd(T x) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(x) * x_sigmoid;
}
"""


swish_fwd = torch.cuda.jiterator._create_jit_fn(swish_fwd_codestring)


class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return swish_bwd(x, dout)


swish = SwishFunction.apply


ACT2FN = {'relu': F.relu, 'sigmoid': sigmoid, 'logsigmoid': logsigmoid, 'silu': swish, 'swish': swish, 'sqrelu': sqrelu, 'gelu': fast_gelu_impl, 'bias_gelu': bias_gelu_impl}


class ShortConvolution(nn.Conv1d):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """

    def __init__(self, hidden_size: 'int', kernel_size: 'int', bias: 'bool'=False, activation: 'Optional[str]'='silu', use_fast_conv1d: 'Optional[bool]'=True):
        super().__init__(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, groups=hidden_size, bias=bias, padding=kernel_size - 1)
        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            assert activation in ['silu', 'swish'], f'Activation `{activation}` not supported yet.'
            self.activation = activation
        if causal_conv1d_fn is None:
            if use_fast_conv1d:
                raise RuntimeError('Please either install `causal-conv1d>=1.4.0` to enable fast causal short convolution CUDA kernel or set `use_fast_conv1d` to False')
            else:
                warnings.warn('The naive Pytorch verison is very slow in practice, please run `pip install causal-conv1d>=1.4.0` to install fast causal short convolution CUDA kernel', category=ImportWarning)
        self.use_fast_conv1d = use_fast_conv1d

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        if not self.use_fast_conv1d:
            s += ', use_fast_conv1d={use_fast_conv1d}'
        return s.format(**self.__dict__)

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, cache: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, seq_idx: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[batch_size, seq_len, hidden_size]`
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[batch_size, hidden_size, kernel_size]`.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[batch_size, hidden_size, kernel_size]`. Default: `False`.
            seq_idx (Optional[torch.Tensor]):
                Sequence index for each token. Used for varlen. Default: `None`.
                Shape: [batch_size, seq_len]
                Suppose a batch consists of two sequences with lengths 3 and 4, seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
        Returns:
            Tensor of shape `[batch_size, seq_len, hidden_size]`.
        """
        batch_size, _, hidden_size = x.shape
        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))
        if output_final_state and cache is None:
            cache = x.new_zeros(batch_size, hidden_size, self.kernel_size[0])
        if cache is not None and x.shape[1] == 1:
            return self.step(x, cache)
        x = rearrange(x, 'b t d -> b d t')
        if cache is not None:
            cache.copy_(F.pad(x, (self.kernel_size[0] - x.shape[-1], 0)))
        if self.use_fast_conv1d:
            x = causal_conv1d_fn(x=x, weight=rearrange(self.weight, 'd 1 w -> d w'), bias=self.bias, activation=self.activation, seq_idx=seq_idx)
        else:
            x = self._conv_forward(x, self.weight, self.bias)[..., :x.shape[-1]]
            if self.activation is not None:
                x = ACT2FN[self.activation](x)
        return rearrange(x, 'b d t -> b t d'), cache

    def step(self, x: 'torch.Tensor', cache: 'torch.Tensor'):
        assert x.shape[1] == 1, 'Only support decoding with 1 token at a time for now'
        x = x.squeeze(1)
        if self.use_fast_conv1d:
            x = causal_conv1d_update(x=x, conv_state=cache, weight=rearrange(self.weight, 'd 1 w -> d w'), bias=self.bias, activation=self.activation)
        else:
            dtype = x.dtype
            cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * rearrange(self.weight, 'd 1 w -> d w'), dim=-1)
            if self.bias is not None:
                x = x + self.bias
            if self.activation is not None:
                x = ACT2FN[self.activation](x)
        return x.unsqueeze(1), cache

    @property
    def state_size(self) ->int:
        return self.hidden_size * self.kernel_size


class ChunkABCFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, s, initial_state, output_final_state):
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT, BC = 64, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NV, NM = triton.cdiv(V, BV), triton.cdiv(M, BM)
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def fwd_pre(s, B, H, T, S):
            z = torch.empty_like(s, dtype=torch.float)
            grid = B * H,
            logcumsumexp_fwd_kernel[grid](s, z, s.stride(1), s.stride(2), s.stride(3), T=T, S=S)
            return z

        def fwd_inner(q, k, v, z, B, H, T, K, V, BT, BK, BV, NT, normk=False, h0=None, ht=None):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            h = q.new_empty(B, H, NT * K, V)
            grid = NV, NK, B * H
            chunk_abc_fwd_kernel_h[grid](k, v, z, h, h0, ht, k.stride(1), k.stride(2), k.stride(3), v.stride(1), v.stride(2), v.stride(3), h.stride(1), h.stride(2), h.stride(3), T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT, NORMK=normk, USE_INITIAL_STATE=h0 is not None, STORE_FINAL_STATE=ht is not None, num_warps=num_warps, num_stages=num_stages)
            return h
        final_state = None
        if output_final_state:
            final_state = q.new_empty(B, H, K, M, dtype=torch.float), q.new_empty(B, H, M, V, dtype=torch.float)
        z = fwd_pre(s, B, H, T, M)
        scale = K ** -0.5
        hk = fwd_inner(q=q, k=k, v=s, z=z, B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT, normk=False, h0=initial_state[0] if initial_state is not None else None, ht=final_state[0] if final_state is not None else None)
        ok1 = torch.empty_like(s)
        Ak = q.new_empty(B, H, T, BT)
        grid = NM, NT, B * H
        chunk_abc_fwd_kernel_K[grid](q, k, z, hk, ok1, Ak, k.stride(1), k.stride(2), k.stride(3), s.stride(1), s.stride(2), s.stride(3), hk.stride(1), hk.stride(2), hk.stride(3), scale=scale, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, num_warps=num_warps, num_stages=num_stages)
        ok0 = torch.empty_like(s)
        grid = NM, NT * NC, B * H
        chunk_abc_fwd_kernel_intra_K[grid](s, z, ok0, Ak, s.stride(1), s.stride(2), s.stride(3), T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC, num_warps=2, num_stages=num_stages)
        ok = ok0.add_(ok1)
        scale = 1.0
        p = torch.empty_like(ok, dtype=torch.float)
        grid = NT, B * H
        softmax_fwd_kernel[grid](ok, p, s.stride(1), s.stride(2), s.stride(3), T=T, S=M, BT=BT)
        qv = p
        scale = 1.0
        hv = fwd_inner(q=qv, k=s, v=v, z=z, B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT, normk=True, h0=initial_state[1] if initial_state is not None else None, ht=final_state[1] if final_state is not None else None)
        Av = q.new_zeros(NM, B, H, T, BT)
        grid = NM, NT * NC * NC, B * H
        chunk_abc_fwd_kernel_intra_V[grid](qv, s, z, Av, s.stride(1), s.stride(2), s.stride(3), scale=scale, T=T, K=M, BT=BT, BC=BC, BK=BM, NC=NC, num_warps=2, num_stages=num_stages)
        Av = Av.sum(0)
        ov = torch.empty_like(v)
        grid = NV, NT, B * H
        chunk_abc_fwd_kernel_V[grid](qv, v, z, hv, ov, Av, s.stride(1), s.stride(2), s.stride(3), v.stride(1), v.stride(2), v.stride(3), hv.stride(1), hv.stride(2), hv.stride(3), scale=scale, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, num_warps=num_warps, num_stages=num_stages)
        ctx.save_for_backward(q, k, v, s, z, ok, p, hk, hv, Av)
        ctx.BT = BT
        return ov, final_state

    @staticmethod
    @contiguous
    def backward(ctx, dov, dht=None):
        q, k, v, s, z, ok, p, hk, hv, Av = ctx.saved_tensors
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT, BC = ctx.BT, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NK, NM = triton.cdiv(K, BK), triton.cdiv(M, BM)
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def bwd_inner(q, z, do, B, H, T, K, V, BT, BK, BV, NT, scale, normk=False):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            dh = q.new_empty(B, H, NT * K, V)
            grid = NK, NV, B * H
            chunk_abc_bwd_kernel_dh[grid](q, z, do, dh, q.stride(1), q.stride(2), q.stride(3), do.stride(1), do.stride(2), do.stride(3), dh.stride(1), dh.stride(2), dh.stride(3), scale=scale, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT, NORMK=normk, num_warps=num_warps, num_stages=num_stages)
            return dh

        def bwd_post(s, z, ss, B, H, T, S, BT, BC, BS, NT, NC, NS):
            doo = torch.empty_like(s)
            grid = NS, B * H
            chunk_abc_bwd_kernel_rcum_inter[grid](s, z, ss, doo, s.stride(1), s.stride(2), s.stride(3), T=T, S=S, BT=BT, BS=BS, NT=NT, num_warps=num_warps, num_stages=num_stages)
            grid = NS, NT * NC, B * H
            chunk_abc_bwd_kernel_rcum_intra[grid](s, z, ss, doo, s.stride(1), s.stride(2), s.stride(3), T=T, S=S, BT=BT, BC=BC, BS=BS, NC=NC, num_warps=num_warps, num_stages=num_stages)
            return doo
        scale = 1.0
        qv = p
        dhv = bwd_inner(qv, z, dov, B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, NT=NT, scale=scale, normk=True)
        dp1 = torch.empty_like(p)
        dsv1 = torch.empty_like(s, dtype=torch.float)
        dv = v.new_empty(NM, *v.shape)
        dAv = q.new_zeros(B, H, T, BT)
        grid = NM, NT, B * H
        chunk_abc_bwd_kernel_V[grid](s, v, z, hv, Av, dov, dhv, dp1, dsv1, dv, dAv, s.stride(1), s.stride(2), s.stride(3), v.stride(1), v.stride(2), v.stride(3), hv.stride(1), hv.stride(2), hv.stride(3), scale=scale, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, num_warps=num_warps, num_stages=num_stages)
        dv = dv.sum(0)
        dp0 = torch.empty_like(p)
        dsv0 = s.new_zeros(s.shape, dtype=torch.float)
        grid = NM, NT * NC, B * H
        chunk_abc_bwd_kernel_intra_V[grid](qv, s, z, dAv, dp0, dsv0, s.stride(1), s.stride(2), s.stride(3), T=T, K=M, BT=BT, BC=BC, BK=BM, NC=NC, num_warps=2, num_stages=num_stages)
        dp = dp1.add_(dp0)
        dsv = dsv1.add_(dsv0)
        dok = torch.empty_like(ok)
        grid = NT, B * H
        softmax_bwd_kernel[grid](p, dp, dok, s.stride(1), s.stride(2), s.stride(3), T=T, S=M, BT=BT)
        scale = K ** -0.5
        dhk = bwd_inner(q, z, dok, B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, NT=NT, scale=scale, normk=False)
        dAk = q.new_zeros(NM, B, H, T, BT)
        grid = NM, NT * NC * NC, B * H
        chunk_abc_bwd_kernel_intra_K[grid](s, z, dok, dAk, s.stride(1), s.stride(2), s.stride(3), scale=scale, T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC, num_warps=2, num_stages=num_stages)
        dAk = dAk.sum(0)
        Ak = q.new_zeros(NK, B, H, T, BT)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dsk1 = s.new_empty(NK, *s.shape, dtype=torch.float)
        grid = NK, NT, B * H
        chunk_abc_bwd_kernel_K[grid](q, k, s, z, hk, Ak, dok, dhk, dq, dk, dsk1, dAk, q.stride(1), q.stride(2), q.stride(3), s.stride(1), s.stride(2), s.stride(3), hk.stride(1), hk.stride(2), hk.stride(3), scale=scale, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, num_warps=num_warps, num_stages=num_stages)
        Ak = Ak.sum(0)
        dsk1 = dsk1.sum(0)
        dsk0 = torch.empty_like(s, dtype=torch.float)
        grid = NM, NT * NC, B * H
        chunk_abc_bwd_kernel_intra_KV[grid](s, z, Ak, dok, dsk0, s.stride(1), s.stride(2), s.stride(3), T=T, V=M, BT=BT, BC=BC, BV=BM, NC=NC, num_warps=2, num_stages=num_stages)
        ds = dsv.add_(dsk1.add_(dsk0))
        ds -= bwd_post(s, z, ok * dok + p * dp, B, H, T, M, BT, BC, BM, NT, NC, NM)
        ds = ds
        return dq, dk, dv, ds, None, None


def chunk_abc(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', initial_state: 'Optional[Tuple[torch.Tensor]]'=None, output_final_state: 'bool'=False, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        s (torch.Tensor):
            slot representations of shape `[B, H, T, M]` if `head_first=True` else `[B, T, H, M]`
        initial_state (Optional[Tuple[torch.Tensor, torch.Tensor]]):
            Initial states of shape `[B, H, K, M]` and `[B, H, M, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, M]` and `[B, H, M, V]`. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, M]` and `[B, H, M, V]` if `output_final_state=True` else `None`.
    """
    if not head_first:
        q, k, v, s = map(lambda x: x.transpose(1, 2), (q, k, v, s))
    o, final_state = ChunkABCFunction.apply(q, k, v, s, initial_state, output_final_state)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state


swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""


swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)


swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""


swiglu_fwd = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)


class SwiGLUFunction(torch.autograd.Function):
    """
    Swish-Gated Linear Unit (SwiGLU) function.

    .. math::
        \\text{SwiGLU}(x, y) = swish(x) * y = \\frac{x}{1 + \\exp(-x)} * y
    """

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_bwd(x, y, dout)


swiglu = SwiGLUFunction.apply


class Attention(nn.Module):

    def __init__(self, hidden_size: 'int'=2048, num_heads: 'int'=32, num_kv_heads: 'Optional[int]'=None, window_size: 'Optional[int]'=None, rope_theta: 'Optional[float]'=10000.0, max_position_embeddings: 'Optional[int]'=None, norm_first: 'bool'=False, norm_eps: 'float'=1e-05, layer_idx: 'int'=None):
        super().__init__()
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.LongTensor]'=None, past_key_values: 'Optional[Cache]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False, **kwargs) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, 'Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] for padding purposes (0 indicating padding). Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed.'
        batch_size, q_len, _ = hidden_states.size()
        if self.norm_first:
            hidden_states = self.norm(hidden_states)
        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', h=self.num_kv_heads)
        cu_seqlens = kwargs.get('cu_seqlens', None)
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset
            if attention_mask is not None:
                seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]).clamp(min=0)
                max_seqlen = q.shape[1] + max(seqlen_offset)
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
        if past_key_values is not None:
            k, v = past_key_values.update(attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)), layer_idx=self.layer_idx, offset=q_len, cache_kwargs=dict(window_size=self.window_size))['attn_state']
            k = rearrange(k, '... (h d) -> ... h d', h=self.num_kv_heads)
            v = rearrange(v, '... (h d) -> ... h d', h=self.num_kv_heads)
        if flash_attn_func is None:
            raise ImportError('Please install Flash Attention via `pip install flash-attn --no-build-isolation` first')
        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(q, k, v, attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, causal=True, window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0))
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(q.squeeze(0), k.squeeze(0), v.squeeze(0), cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, causal=True, window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0)).unsqueeze(0)
        else:
            o = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0))
        o = o.reshape(batch_size, q_len, self.hidden_size)
        o = self.o_proj(o)
        if not output_attentions:
            attentions = None
        return o, attentions, past_key_values

    def _upad_input(self, q, k, v, attention_mask, q_len):
        seqlens = attention_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)
        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


def checkpoint(fn):

    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return wrapper


@checkpoint
def flatten_diag_outer_product_off1(x, y):
    z = torch.einsum('...i,...j->...ij', x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indices2 = torch.arange(0, N)
    return z[..., indicies[0], indicies[1]], z[..., indices2, indices2]


def chunk_local_cumsum_scalar(g: 'torch.Tensor', chunk_size: 'int', reverse: 'bool'=False, offsets: 'Optional[torch.Tensor]'=None, indices: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True, output_dtype: 'Optional[torch.dtype]'=torch.float) ->torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    if offsets is not None:
        B = len(offsets) - 1
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), 'chunk_size must be a power of 2'
    BT = chunk_size
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.stack([offsets.new_full((n,), i), offsets.new_tensor(range(n))], 1) for i, n in enumerate(triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist())])
        NT = len(indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = NT, B * H
    chunk_local_cumsum_scalar_kernel[grid](g_org, g, offsets, indices, T=T, H=H, BT=BT, HEAD_FIRST=head_first, REVERSE=reverse)
    return g


def chunk_local_cumsum_vector(g: 'torch.Tensor', chunk_size: 'int', reverse: 'bool'=False, offsets: 'Optional[torch.Tensor]'=None, indices: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True, output_dtype: 'Optional[torch.dtype]'=torch.float) ->torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), 'chunk_size must be a power of 2'
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.stack([offsets.new_full((n,), i), offsets.new_tensor(range(n))], 1) for i, n in enumerate(triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist())])
        NT = len(indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return triton.cdiv(meta['S'], meta['BS']), NT, B * H
    chunk_local_cumsum_vector_kernel[grid](g_org, g, offsets, indices, T=T, H=H, S=S, BT=BT, HEAD_FIRST=head_first, REVERSE=reverse)
    return g


@contiguous
def chunk_local_cumsum(g: 'torch.Tensor', chunk_size: 'int', reverse: 'bool'=False, offsets: 'Optional[torch.Tensor]'=None, indices: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True, output_dtype: 'Optional[torch.dtype]'=torch.float) ->torch.Tensor:
    if offsets is not None:
        assert g.shape[0] == 1, 'Only batch size 1 is supported when offsets are provided'
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(g, chunk_size, reverse, offsets, indices, head_first, output_dtype)
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(g, chunk_size, reverse, offsets, indices, head_first, output_dtype)
    else:
        raise ValueError(f'Unsupported input shape {g.shape}. which should be (B, H, T, dim) if `head_first=True` or (batch_size, num_heads, seq_len) otherwise')


def chunk_bwd_dh(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', gk: 'torch.Tensor', gv: 'torch.Tensor', do: 'torch.Tensor', h0: 'torch.Tensor', dht: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.Tensor]'=None, split_offsets: 'Optional[torch.Tensor]'=None, split_indices: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64, split_size: 'int'=256, states_in_fp32: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[2]
    S, BT = max(chunk_size, min(split_size, triton.next_power_of_2(T))), chunk_size
    assert S % BT == 0, f'The `split_size` (got {S}) must be a multiple of `chunk_size` {BT}'
    if offsets is None:
        N = B
        NS = N * triton.cdiv(T, S)
    else:
        N = len(offsets) - 1
        NS = split_offsets[-1]
    NG = HQ // H
    dhs = q.new_empty(NS, HQ, K, V, dtype=torch.float)
    dhr = q.new_empty(NS, HQ, K, V, dtype=torch.float if states_in_fp32 else k.dtype)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta):
        return triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), NS * HQ
    chunk_bwd_kernel_dh_split[grid](q=q, g=g, gk=gk, gv=gv, do=do, dht=dht, dhs=dhs, dhr=dhr, dh0=dh0, offsets=offsets, split_indices=split_indices, scale=scale, T=T, S=S, HQ=HQ, H=H, K=K, V=V, BT=BT, NG=NG, USE_G=g is not None, USE_GK=gk is not None, USE_GV=gv is not None, HEAD_FIRST=head_first)

    def grid(meta):
        return triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * HQ
    chunk_bwd_kernel_dh_reduction[grid](g=g, gk=gk, gv=gv, dhs=dhs, dhr=dhr, dh0=dh0, offsets=offsets, split_offsets=split_offsets, T=T, S=S, HQ=HQ, H=H, K=K, V=V, BT=BT, NG=NG, USE_G=g is not None, USE_GK=gk is not None, USE_GV=gv is not None, HEAD_FIRST=head_first)
    return dhr, dh0


def chunk_bwd_dqkwg(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', do: 'torch.Tensor', h: 'torch.Tensor', dh: 'torch.Tensor', dv: 'Optional[torch.Tensor]'=None, w: 'Optional[torch.Tensor]'=None, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, chunk_size: 'int'=64, scale: 'float'=1.0, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None
    grid = NK, NT, B * H
    chunk_bwd_kernel_dqkwg[grid](q=q, k=k, v=v, h=h, g=g, do=do, dh=dh, dv=dv, w=w, dw=dw, dq=dq, dk=dk, dg=dg, offsets=offsets, indices=indices, scale=scale, B=B, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, HEAD_FIRST=head_first)
    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg


def chunk_bwd_dv(q: 'torch.Tensor', k: 'torch.Tensor', g: 'torch.Tensor', do: 'torch.Tensor', dh: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->torch.Tensor:
    if head_first:
        B, H, T, K, V = *k.shape, do.shape[-1]
    else:
        B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 128)
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NV = triton.cdiv(V, BV)
    dv = torch.empty_like(do)
    grid = NV, NT, B * H
    chunk_bwd_kernel_dv[grid](q, k, g, do, dv, dh, offsets, indices, scale, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return dv


def chunk_fwd_h(k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', gk: 'torch.Tensor', gv: 'torch.Tensor', h0: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, split_offsets: 'Optional[torch.LongTensor]'=None, split_indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64, split_size: 'int'=256, states_in_fp32: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    S, BT = split_size, chunk_size
    assert S % BT == 0, f'The `split_size` (got {S}) must be a multiple of `chunk_size` {BT}'
    if offsets is None:
        N = B
        NS = N * triton.cdiv(T, S)
    else:
        N = len(offsets) - 1
        NS = split_offsets[-1]
    hs = k.new_empty(NS, H, K, V, dtype=torch.float)
    hr = k.new_empty(NS, H, K, V, dtype=torch.float if states_in_fp32 else k.dtype)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None

    def grid(meta):
        return triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), NS * H
    chunk_fwd_kernel_h_split[grid](k=k, v=v, g=g, gk=gk, gv=gv, hs=hs, hr=hr, h0=h0, ht=ht, offsets=offsets, split_indices=split_indices, T=T, S=S, H=H, K=K, V=V, BT=BT, USE_G=g is not None, USE_GK=gk is not None, USE_GV=gv is not None, HEAD_FIRST=head_first)

    def grid(meta):
        return triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H
    chunk_fwd_kernel_h_reduction[grid](g=g, gk=gk, gv=gv, hs=hs, hr=hr, ht=ht, offsets=offsets, split_offsets=split_offsets, T=T, S=S, H=H, K=K, V=V, BT=BT, USE_G=g is not None, USE_GK=gk is not None, USE_GV=gv is not None, HEAD_FIRST=head_first)
    return hr, ht


def chunk_simple_gla_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', initial_state: 'torch.Tensor', do: 'torch.Tensor', dht: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, _ = chunk_fwd_h(k=k, v=v, g=g, gk=None, gv=None, h0=initial_state, output_final_state=False, states_in_fp32=True, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    dh, dh0 = chunk_bwd_dh(q=q, k=k, v=v, g=g, gk=None, gv=None, do=do, h0=initial_state, dht=dht, scale=scale, states_in_fp32=True, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    dq, dk, _, dg = chunk_bwd_dqkwg(q=q, k=k, v=v, g=g, h=h, do=do, dh=dh, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    dv = chunk_bwd_dv(q=q, k=k, g=g, do=do, dh=dh, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    return dq, dk, dv, dg, dh0


def chunk_fwd_o(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', h: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None, scale: 'Optional[float]'=None, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->torch.Tensor:
    if head_first:
        B, H, T, K, V = *q.shape, v.shape[-1]
    else:
        B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = k.shape[-1] ** -0.5
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    o = torch.empty_like(v)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_fwd_kernel_o[grid](q, k, v, h, g, o, offsets, indices, scale, T=T, H=H, K=K, V=V, BT=BT, HEAD_FIRST=head_first)
    return o


def chunk_simple_gla_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = chunk_local_cumsum(g, chunk_size, offsets=offsets, head_first=head_first) if g is not None else None
    h, ht = chunk_fwd_h(k=k, v=v, g=g, gk=None, gv=None, h0=initial_state, output_final_state=output_final_state, states_in_fp32=False, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    o = chunk_fwd_o(q=q, k=k, v=v, g=g, h=h, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    return g, o, ht


def chunk_simple_gla(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.simple_gla import chunk_simple_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, device='cuda'))
        >>> o, ht = chunk_simple_gla(q, k, v, g,
                                     initial_state=None,
                                     output_final_state=True,
                                     head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_simple_gla(q, k, v, g,
                                             initial_state=None,
                                             output_final_state=True,
                                             cu_seqlens=cu_seqlens,
                                             head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkSimpleGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state, cu_seqlens, head_first)
    return o, final_state


@torch.jit.script
def normalize_output(q, k, o):
    k = k.cumsum(-2)
    z = (q * k).sum(-1, keepdim=True)
    return o / (z + 1e-10)


def chunk_linear_attn(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, normalize: 'bool'=True, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        scale (Optional[int]):
            Scale factor for the linear attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        normalize (bool):
            Whether to normalize the output. Default: `True`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = chunk_simple_gla(q=q, k=k, v=v, scale=scale, g=None, initial_state=initial_state, output_final_state=output_final_state, head_first=head_first)
    if normalize:
        o = normalize_output(q * scale, k, o)
    return o, final_state


def fused_chunk_linear_attn(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, normalize: 'bool'=True, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        scale (Optional[int]):
            Scale factor for linear attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        normalize (bool):
            Whether to normalize the output. Default: `True`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, final_state = FusedChunkLinearAttentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    if normalize:
        o = normalize_output(q * scale, k, o)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state


def parallel_based(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', scale: 'Optional[float]'=None, use_norm: 'bool'=True, head_first: 'bool'=True):
    assert q.shape[-1] <= 128, 'only support feature dim up to 128'
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, z = triton_parallel_based(q, k, v, scale)
    if use_norm:
        o = o / (z[..., None] + 1e-06)
    if not head_first:
        o = o.transpose(1, 2)
    return o


class BasedLinearAttention(nn.Module):

    def __init__(self, hidden_size: 'int', feature_dim: 'int'=16, num_key_value_heads: 'int'=12, num_heads: 'int'=12, feature_name: 'str'='taylor_exp', eps: 'float'=1e-12, causal: 'bool'=True, mode: 'str'='parallel'):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_key_value_heads
        self.causal = causal
        self.q_proj = nn.Linear(self.hidden_size, self.feature_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.feature_dim * self.num_heads, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Identity()
        self.feature_map = TaylorFeatureMap(feature_dim)
        self.eps = eps
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: 'nn.Module'):
        if getattr(module, '_is_hf_initialized', False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, hidden_states: 'torch.Tensor', **kwargs):
        mode = self.mode
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q, k, v = map(lambda x: rearrange(x, '... (h d) -> ... h d', h=self.num_heads), [q, k, v])
        if mode == 'fused_chunk':
            q, k = self.feature_map(q), self.feature_map(k)
            o, _ = fused_chunk_linear_attn(q, k, v, normalize=True, scale=1, head_first=False)
        elif mode == 'chunk':
            q, k = self.feature_map(q), self.feature_map(k)
            o, _ = chunk_linear_attn(q, k, v, normalize=True, scale=1, head_first=False)
        elif mode == 'parallel':
            assert q.shape[-1] <= 128
            o = parallel_based(q, k, v, scale=1, use_norm=True, head_first=False)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        o = self.dropout(o)
        return o

    def forward_reference(self, hidden_states: 'torch.Tensor', filters: 'torch.Tensor'=None, *args, **kwargs):
        """
        x (torch.Tensor): tensor of shape (b, d, t)
        y (torch.Tensor): tensor of shape (b, d, t)
        """
        b, t, _ = hidden_states.size()
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q = q.view(b, t, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, t, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        if self.causal:
            y = (q * (k * v).cumsum(2)).sum(-1) / ((q * k.cumsum(2)).sum(-1) + self.eps)
        else:
            y = (q * (k * v).sum(2, True)).sum(-1) / ((q * k.sum(2, True)).sum(-1) + self.eps)
        y = rearrange(y, 'b h t d -> b t (h d)')
        y = self.o_proj(y)
        y = self.dropout(y)
        return y


def activation_quant(x):
    """
    Per-token quantization to 8 bits. No grouping is needed for quantization.

    Args:
        x: An activation tensor with shape [n, d].

    Returns:
        A quantized activation tensor with shape [n, d].
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-05)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-05)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class BitLinear(nn.Linear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=False, norm_eps: 'float'=1e-08):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features, eps=norm_eps)

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}({super().extra_repr()}, norm_eps={self.norm.eps})'

    def forward(self, x):
        """
        Overrides the forward pass to include quantization.

        Args:
            x: An input tensor with shape [n, d].

        Returns:
            An output tensor with shape [n, d].
        """
        w = self.weight
        x_norm = self.norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y


def layer_norm_fwd_quant(x: 'torch.Tensor', weight: 'torch.Tensor', bias: 'torch.Tensor', eps: 'float', residual: 'torch.Tensor'=None, out_dtype: 'torch.dtype'=None, residual_dtype: 'torch.dtype'=None, is_rms_norm: 'bool'=False):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or residual_dtype is not None and residual_dtype != x.dtype:
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device='cuda') if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.device(x.device.index):
        layer_norm_fwd_kernel_quant[M,](x, y, weight, bias, residual, residual_out, mean, rstd, x.stride(0), y.stride(0), residual.stride(0) if residual is not None else 0, residual_out.stride(0) if residual_out is not None else 0, N, eps, is_rms_norm, BLOCK_N, residual is not None, residual_out is not None, weight is not None, bias is not None)
    return y, mean, rstd, residual_out if residual_out is not None else x


class LayerNormLinearQuantFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, norm_weight, norm_bias, linear_weight, linear_bias, residual=None, eps=1e-06, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = residual.dtype if residual is not None else torch.float32 if residual_in_fp32 else None
        y, mean, rstd, residual_out = layer_norm_fwd_quant(x, norm_weight, norm_bias, eps, residual, out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(), residual_dtype=residual_dtype, is_rms_norm=is_rms_norm)
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = weight_quant(linear_weight)
        linear_bias = linear_bias if linear_bias is not None else None
        out = F.linear(y, linear_weight, linear_bias)
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dout, *args):
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(dy, x, norm_weight, norm_bias, ctx.eps, mean, rstd, dresidual, ctx.has_residual, ctx.is_rms_norm, x_dtype=ctx.x_dtype, recompute_output=True)
        dlinear_weight = torch.einsum('bo,bi->oi', dout, y)
        return dx.reshape(ctx.x_shape_og), dnorm_weight, dnorm_bias, dlinear_weight, dlinear_bias, dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None, None, None, None, None


def layer_norm_linear_quant_fn(x, norm_weight, norm_bias, linear_weight, linear_bias, residual=None, eps=1e-06, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
    return LayerNormLinearQuantFn.apply(x, norm_weight, norm_bias, linear_weight, linear_bias, residual, eps, prenorm, residual_in_fp32, is_rms_norm)


class FusedBitLinear(BitLinear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(self, in_features, out_features, bias=False):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        super(FusedBitLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return layer_norm_linear_quant_fn(x, self.norm.weight, self.norm.bias, self.weight, self.bias, is_rms_norm=True)


class BitAttention(nn.Module):

    def __init__(self, hidden_size: 'int'=2048, num_heads: 'int'=32, num_kv_heads: 'Optional[int]'=None, window_size: 'Optional[int]'=None, rope_theta: 'Optional[float]'=10000.0, max_position_embeddings: 'Optional[int]'=None, norm_first: 'bool'=False, norm_eps: 'float'=1e-05, layer_idx: 'int'=None):
        super().__init__()
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.norm_first = norm_first
        self.layer_idx = layer_idx
        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.q_proj = FusedBitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = FusedBitLinear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = FusedBitLinear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = FusedBitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.LongTensor]'=None, past_key_values: 'Optional[Cache]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False, **kwargs) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, 'Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] for padding purposes (0 indicating padding). Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed.'
        batch_size, q_len, _ = hidden_states.size()
        if self.norm_first:
            hidden_states = self.norm(hidden_states)
        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', h=self.num_kv_heads)
        cu_seqlens = kwargs.get('cu_seqlens', None)
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset
            if attention_mask is not None:
                seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]).clamp(min=0)
                max_seqlen = q.shape[1] + max(seqlen_offset)
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
        if past_key_values is not None:
            k, v = past_key_values.update(attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)), layer_idx=self.layer_idx, offset=q_len, cache_kwargs=dict(window_size=self.window_size))['attn_state']
            k = rearrange(k, '... (h d) -> ... h d', h=self.num_kv_heads)
            v = rearrange(v, '... (h d) -> ... h d', h=self.num_kv_heads)
        if flash_attn_func is None:
            raise ImportError('Please install Flash Attention via `pip install flash-attn --no-build-isolation` first')
        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(q, k, v, attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, causal=True, window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0))
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(q.squeeze(0), k.squeeze(0), v.squeeze(0), cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, causal=True, window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0)).unsqueeze(0)
        else:
            o = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0))
        o = o.reshape(batch_size, q_len, self.hidden_size)
        o = self.o_proj(o)
        if not output_attentions:
            attentions = None
        return o, attentions, past_key_values

    def _upad_input(self, q, k, v, attention_mask, q_len):
        seqlens = attention_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)
        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


def bwd_prepare_wy_repr(k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', beta: 'torch.Tensor', Aw: 'torch.Tensor', Au: 'torch.Tensor', dw: 'torch.Tensor', du: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]', indices: 'Optional[torch.LongTensor]', head_first: 'bool', chunk_size: 'int') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    dg = torch.empty_like(g)
    bwd_prepare_wy_repr_kernel[NT, B * H](k=k, v=v, beta=beta, g=g, Aw=Aw, Au=Au, dw=dw, du=du, dk=dk, dv=dv, dbeta=dbeta, dg=dg, offsets=offsets, indices=indices, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return dk, dv, dbeta, dg


def chunk_bwd_dv_local(q: 'torch.Tensor', k: 'torch.Tensor', g: 'torch.Tensor', do: 'torch.Tensor', dh: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->torch.Tensor:
    if head_first:
        B, H, T, K, V = *k.shape, do.shape[-1]
    else:
        B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 128)
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    dv = torch.empty_like(do)
    grid = NT, B * H
    chunk_bwd_kernel_dv_local[grid](q, k, g, do, dv, offsets, indices, scale, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return dv


def tensor_cache(fn: 'Callable[..., torch.Tensor]') ->Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: 'Optional[Tuple]' = None
    last_kwargs: 'Optional[Dict]' = None
    last_result: 'Any' = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) ->Any:
        nonlocal last_args, last_kwargs, last_result
        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result
        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result
    return wrapper


@tensor_cache
def prepare_chunk_offsets(offsets: 'torch.Tensor', chunk_size: 'int') ->Tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], chunk_size)]).cumsum(-1)


def chunk_gated_delta_rule_bwd_dhu(q: 'torch.Tensor', k: 'torch.Tensor', w: 'torch.Tensor', g: 'torch.Tensor', h0: 'torch.Tensor', dht: 'Optional[torch.Tensor]', do: 'torch.Tensor', dv: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *q.shape, do.shape[-1]
    else:
        B, T, H, K, V = *q.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, 'current kernel does not support head dimension being larger than 256.'
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64 if K <= 128 else 32
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    if head_first:
        dh = q.new_empty(B, H, NT, K, V)
    else:
        dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)
    grid = NK, NV, N * H
    chunk_gated_delta_rule_bwd_kernel_dhu[grid](q=q, k=k, d=w, g=g, dht=dht, dh0=dh0, do=do, dh=dh, dv=dv, dv2=dv2, offsets=offsets, chunk_offsets=chunk_offsets, scale=scale, T=T, H=H, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return dh, dh0, dv2


def chunk_gated_delta_rule_fwd_h(k: 'torch.Tensor', w: 'torch.Tensor', u: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, u.shape[-1]
    else:
        B, T, H, K, V = *k.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, 'current kernel does not support head dimension larger than 256.'
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    if head_first:
        h = k.new_empty(B, H, NT, K, V)
    else:
        h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)
    grid = NK, NV, N * H
    chunk_gated_delta_rule_fwd_kernel_h[grid](k=k, v=u, d=w, v_new=v_new, g=g, h=h, h0=initial_state, ht=final_state, offsets=offsets, chunk_offsets=chunk_offsets, T=T, H=H, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT, HEAD_FIRST=head_first)
    return h, v_new, final_state


def fwd_recompute_w_u(k: 'torch.Tensor', v: 'torch.Tensor', beta: 'torch.Tensor', Aw: 'torch.Tensor', Au: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]', indices: 'Optional[torch.LongTensor]', head_first: 'bool', chunk_size: 'int') ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    u = torch.empty_like(v)
    w = torch.empty_like(k)
    fwd_recompute_w_u_kernel[NT, B * H](k=k, v=v, beta=beta, w=w, u=u, Aw=Aw, Au=Au, offsets=offsets, indices=indices, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return w, u


def chunk_delta_rule_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', beta: 'torch.Tensor', A: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', do: 'torch.Tensor', dht: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u = fwd_recompute_w_u(k=k, v=v, beta=beta, A=A, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=initial_state, output_final_state=False, offsets=offsets, head_first=head_first, chunk_size=BT)
    dv = chunk_bwd_dv_local(q=q, k=k, do=do, g=None, dh=None, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, g=None, h0=initial_state, dht=dht, do=do, dv=dv, scale=scale, offsets=offsets, head_first=head_first, chunk_size=BT)
    dq, dk, dw, _ = chunk_bwd_dqkwg(q=q, k=k, v=v_new, h=h, w=w, dv=dv, do=do, dh=dh, g=None, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dk2, dv, db = bwd_prepare_wy_repr(k=k, v=v, beta=beta, A=A, dw=dw, du=dv, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dk.add_(dk2)
    return dq, dk, dv, db, dh0


def fwd_wu(a: 'torch.Tensor', v: 'torch.Tensor', k: 'torch.Tensor', A: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]', indices: 'Optional[torch.LongTensor]', head_first: 'bool', chunk_size: 'int') ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *a.shape, v.shape[-1]
    else:
        B, T, H, K, V = *a.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    u = torch.empty_like(v)
    w = torch.empty_like(a)
    fwd_wu_kernel[NT, B * H](a=a, v=v, w=w, u=u, A=A, k=k, offsets=offsets, indices=indices, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return w, u


def fwd_prepare_wy_repr(a: 'torch.Tensor', b: 'torch.Tensor', v: 'torch.Tensor', k: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]', indices: 'Optional[torch.LongTensor]', head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K = a.shape
    else:
        B, T, H, K = a.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    BC = min(BT, 32)
    BK = min(triton.next_power_of_2(K), 64)
    A = torch.empty(B, *((H, T) if head_first else (T, H)), BT, device=a.device, dtype=a.dtype)
    fwd_fn = fwd_prepare_wy_repr_kernel_chunk64 if BT == 64 else fwd_prepare_wy_repr_kernel_chunk32
    fwd_fn[NT, B * H](a=a, b=b, A=A, offsets=offsets, indices=indices, T=T, H=H, K=K, BT=BT, BK=BK, BC=BC, HEAD_FIRST=head_first)
    w, u = fwd_wu(a=a, v=v, k=k, A=A, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    return w, u, A


def chunk_delta_rule_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', beta: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u, A = fwd_prepare_wy_repr(k=k, v=v, beta=beta, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=initial_state, output_final_state=output_final_state, offsets=offsets, head_first=head_first, chunk_size=BT)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    return o, A, final_state


def l2norm_bwd(x: 'torch.Tensor', dy: 'torch.Tensor', eps: 'float'=1e-05):
    x_shape_og = x.shape
    x = x.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    assert dy.shape == x.shape
    dx = torch.empty_like(x)
    N = x.shape[-1]
    M = x.shape[0]
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.device(x.device.index):
        l2norm_bwd_kernel[M,](x, dy, dx, x.stride(0), N, eps, BLOCK_N)
    return dx.reshape(x_shape_og)


def l2norm_fwd(x: 'torch.Tensor', eps: 'float'=1e-06, output_dtype: 'Optional[torch.dtype]'=None):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
        M, N = x.shape
    assert x.stride(-1) == 1
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    N = x.shape[-1]
    M = x.shape[0]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.device(x.device.index):
        l2norm_fwd_kernel[M,](x, y, x.stride(0), N, eps, BLOCK_N)
    return y.reshape(x_shape_og)


@tensor_cache
def prepare_chunk_indices(offsets: 'torch.LongTensor', chunk_size: 'int') ->Tuple[torch.LongTensor]:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
    indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
    return indices


def chunk_delta_rule(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', beta: 'torch.Tensor', scale: 'float'=None, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, use_qk_l2norm_in_kernel: 'bool'=False):
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        beta (torch.Tensor):
            betas of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use qk l2norm within the kernel for saving GPU memory.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.delta_rule import chunk_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_delta_rule(q, k, v, beta,
                                     initial_state=h0,
                                     output_final_state=True,
                                     head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_delta_rule(q, k, v, beta,
                                             initial_state=h0,
                                             output_final_state=True,
                                             cu_seqlens=cu_seqlens,
                                             head_first=False)
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, 'ChunkDeltaRuleFunction does not support float32. Please use bfloat16.'
    assert len(beta.shape) == 3, 'beta must be of shape (batch size, num of head, seq len).'
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDeltaRuleFunction.apply(q, k, v, beta, scale, initial_state, output_final_state, cu_seqlens, head_first, use_qk_l2norm_in_kernel)
    return o, final_state


def elu_p1(x):
    return F.elu(x, 1.0, False) + 1.0


def fused_recurrent_gated_delta_rule_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', beta: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported yet'
    num_stages = 3
    num_warps = 1
    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        final_state = None
    grid = NV, NK, N * H
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](q=q, k=k, v=v, g=g, beta=beta, o=o, h0=initial_state, ht=final_state, offsets=offsets, scale=scale, B=B, T=T, H=H, K=K, V=V, BK=BK, BV=BV, IS_BETA_HEADWISE=beta.ndim == v.ndim, HEAD_FIRST=head_first, num_warps=num_warps, num_stages=num_stages)
    o = o.squeeze(0)
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', beta: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, use_qk_l2norm_in_kernel: 'bool'=False):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        o, final_state = fused_recurrent_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state, offsets=offsets, head_first=head_first)
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet and we do not have plans to implement it because we haven't figured out how to compute dg without materializing the full hidden states for all time steps.")


def fused_recurrent_delta_rule(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', beta: 'torch.Tensor'=None, scale: 'float'=None, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, use_qk_l2norm_in_kernel: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        beta (torch.Tensor):
             betas of shape `[B, H, T]` if `head_first=True` else `(B, T, H)`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.delta_rule import fused_recurrent_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> beta = torch.rand(B, T, H, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_delta_rule(q, k, v, beta,
                                               initial_state=h0,
                                               output_final_state=True,
                                               head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_delta_rule(q, k, v, beta,
                                                       initial_state=h0,
                                                       output_final_state=True,
                                                       cu_seqlens=cu_seqlens,
                                                       head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, 'scale must be positive'
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(q, k, v, beta, scale, initial_state, output_final_state, cu_seqlens, head_first, use_qk_l2norm_in_kernel)
    return o, final_state


def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


def chunk_gated_delta_rule_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', beta: 'torch.Tensor', Aw: 'torch.Tensor', Au: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', do: 'torch.Tensor', dht: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u = fwd_recompute_w_u(k=k, v=v, beta=beta, Aw=Aw, Au=Au, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=initial_state, output_final_state=False, offsets=offsets, head_first=head_first, chunk_size=BT)
    dv = chunk_bwd_dv_local(q=q, k=k, g=g, do=do, dh=None, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, g=g, h0=initial_state, dht=dht, do=do, dv=dv, scale=scale, offsets=offsets, head_first=head_first, chunk_size=BT)
    dq, dk, dw, dg = chunk_bwd_dqkwg(q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dk2, dv, db, dg2 = bwd_prepare_wy_repr(k=k, v=v, beta=beta, g=g, Aw=Aw, Au=Au, dw=dw, du=dv, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dk.add_(dk2)
    dg.add_(dg2)
    assert dg.dtype == torch.float32, 'dg should be fp32'
    dg = chunk_local_cumsum(dg, chunk_size, reverse=True, offsets=offsets, head_first=head_first)
    return dq, dk, dv, db, dg, dh0


def chunk_gated_delta_rule_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', beta: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    BT = chunk_size
    g = chunk_local_cumsum(g, chunk_size, offsets=offsets, head_first=head_first)
    w, u, Aw, Au = fwd_prepare_wy_repr(k=k, v=v, beta=beta, g=g, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=initial_state, output_final_state=output_final_state, offsets=offsets, head_first=head_first, chunk_size=BT)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    return g, o, Aw, Au, final_state


def chunk_gated_delta_rule(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', beta: 'torch.Tensor', scale: 'float'=None, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, use_qk_l2norm_in_kernel: 'bool'=False):
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(q, k, v, g, beta,
                                           initial_state=h0,
                                           output_final_state=True,
                                           head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(q, k, v, g, beta,
                                                   initial_state=h0,
                                                   output_final_state=True,
                                                   cu_seqlens=cu_seqlens,
                                                   head_first=False)
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, 'ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16.'
    assert len(beta.shape) == 3, 'beta must be of shape [B, H, T] if head_first=True, or [B, T, H] if head_first=False.'
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, 'Scale must be positive.'
    o, final_state = ChunkGatedDeltaRuleFunction.apply(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, head_first, use_qk_l2norm_in_kernel)
    return o, final_state


def fused_recurrent_gated_delta_rule(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', beta: 'torch.Tensor'=None, scale: 'float'=None, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, use_qk_l2norm_in_kernel: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
             g (decays) of shape `[B, H, T]` if `head_first=True` else `(B, T, H)`.
        beta (torch.Tensor):
             betas of shape `[B, H, T]` if `head_first=True` else `(B, T, H)`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, H, device='cuda'))
        >>> beta = torch.rand(B, T, H, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(q, k, v, g, beta,
                                               initial_state=h0,
                                               output_final_state=True,
                                               head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(q, k, v, g, beta,
                                                             initial_state=h0,
                                                             output_final_state=True,
                                                             cu_seqlens=cu_seqlens,
                                                             head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, 'scale must be positive'
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, head_first, use_qk_l2norm_in_kernel)
    return o, final_state


def chunk_gla_bwd_dA(v: 'torch.Tensor', do: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, V = v.shape
    else:
        B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BV = min(64, triton.next_power_of_2(V))
    dA = v.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=torch.float)
    grid = NT, B * H
    chunk_gla_bwd_kernel_dA[grid](v, do, dA, offsets, indices, scale, T=T, H=H, V=V, BT=BT, BV=BV, HEAD_FIRST=head_first)
    return dA


def chunk_gla_bwd_dqk_intra(q: 'torch.Tensor', k: 'torch.Tensor', g: 'torch.Tensor', dA: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K = q.shape
    else:
        B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    grid = NK, NT * NC, B * H
    chunk_gla_bwd_kernel_intra[grid](q, k, g, dA, dq, dk, offsets, indices, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC, HEAD_FIRST=head_first)
    return dq, dk


def chunk_gla_bwd_dqkg(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', h: 'torch.Tensor', g: 'torch.Tensor', do: 'torch.Tensor', dh: 'torch.Tensor', dq: 'torch.Tensor', dk: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    dg = torch.empty_like(g)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)

    def grid(meta):
        return triton.cdiv(K, meta['BK']), NT, B * H
    chunk_gla_bwd_kernel_inter[grid](q, k, v, h, g, do, dh, dq, dk, dq2, dk2, dg, offsets, indices, scale, T=T, H=H, K=K, V=V, BT=BT, HEAD_FIRST=head_first)
    return dq2, dk2, dg


def chunk_gla_bwd_dv(k: 'torch.Tensor', g: 'torch.Tensor', A: 'torch.Tensor', do: 'torch.Tensor', dh: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K, V = *k.shape, do.shape[-1]
    else:
        B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    dv = torch.empty_like(do)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_gla_bwd_kernel_dv[grid](k, g, A, do, dh, dv, offsets, indices, T=T, H=H, K=K, V=V, BT=BT, HEAD_FIRST=head_first)
    return dv


def chunk_gla_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', g_cumsum: 'Optional[torch.Tensor]', scale: 'float', initial_state: 'torch.Tensor', h: 'torch.Tensor', A: 'torch.Tensor', do: 'torch.Tensor', dht: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, BT, offsets=offsets, head_first=head_first)
    if h is None:
        h, _ = chunk_fwd_h(k=k, v=v, g=None, gk=g_cumsum, gv=None, h0=initial_state, output_final_state=False, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT, states_in_fp32=True)
    dh, dh0 = chunk_bwd_dh(q=q, k=k, v=v, g=None, gk=g_cumsum, gv=None, do=do, h0=initial_state, dht=dht, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT, states_in_fp32=True)
    dv = chunk_gla_bwd_dv(k=k, g=g_cumsum, A=A, do=do, dh=dh, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dA = chunk_gla_bwd_dA(v=v, do=do, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dq, dk = chunk_gla_bwd_dqk_intra(q=q, k=k, g=g_cumsum, dA=dA, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    dq, dk, dg = chunk_gla_bwd_dqkg(q=q, k=k, v=v, h=h, g=g_cumsum, do=do, dh=dh, dq=dq, dk=dk, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    return dq, dk, dv, dg, dh0


def chunk_gla_fwd_intra_gk(q: 'torch.Tensor', k: 'torch.Tensor', g: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K = k.shape
    else:
        B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    A = q.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=torch.float)
    grid = NT, NC * NC, B * H
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](q, k, g, A, offsets, indices, scale, T=T, H=H, K=K, BT=BT, BC=BC, NC=NC, HEAD_FIRST=head_first)
    grid = NT, NC, B * H
    if K <= 256:
        BK = triton.next_power_of_2(K)
        chunk_gla_fwd_A_kernel_intra_sub_intra[grid](q, k, g, A, offsets, indices, scale, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, HEAD_FIRST=head_first)
    else:
        BK = min(128, triton.next_power_of_2(K))
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, *((H, T) if head_first else (T, H)), BC, dtype=torch.float)
        grid = NK, NT * NC, B * H
        chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](q, k, g, A_intra, offsets, indices, scale, B=B, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC, HEAD_FIRST=head_first)
        grid = NT, NC, B * H
        chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](A_intra, A, offsets, indices, B=B, T=T, H=H, BT=BT, BC=BC, NK=NK, HEAD_FIRST=head_first)
    return A


def chunk_gla_fwd_o_gk(q: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', A: 'torch.Tensor', h: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K, V = *q.shape, v.shape[-1]
    else:
        B, T, H, K, V = *q.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    o = torch.empty_like(v)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_gla_fwd_kernel_o[grid](q, v, g, h, o, A, offsets, indices, scale, T=T, H=H, K=K, V=V, BT=BT, HEAD_FIRST=head_first)
    return o


def chunk_gla_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', g_cumsum: 'Optional[torch.Tensor]', scale: 'float', initial_state: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, BT, offsets=offsets, head_first=head_first)
    h, ht = chunk_fwd_h(k=k, v=v, g=None, gk=g_cumsum, gv=None, h0=initial_state, output_final_state=output_final_state, states_in_fp32=False, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    A = chunk_gla_fwd_intra_gk(q=q, k=k, g=g_cumsum, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    o = chunk_gla_fwd_o_gk(q=q, v=v, g=g_cumsum, A=A, h=h, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    return g_cumsum, A, h, ht, o


class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state, offsets, head_first):
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))
        indices = prepare_chunk_indices(offsets, chunk_size) if offsets is not None else None
        g_cumsum, A, h, ht, o = chunk_gla_fwd(q=q, k=k, v=v, g=g, g_cumsum=None, scale=scale, initial_state=initial_state, output_final_state=output_final_state, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
        if g.dtype != torch.float:
            g_cumsum = None
        else:
            g = None
        ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        return o, ht

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, g, g_cumsum, initial_state, A = ctx.saved_tensors
        chunk_size, scale, offsets, indices, head_first = ctx.chunk_size, ctx.scale, ctx.offsets, ctx.indices, ctx.head_first
        dq, dk, dv, dg, dh0 = chunk_gla_bwd(q=q, k=k, v=v, g=g, g_cumsum=g_cumsum, scale=scale, h=None, A=A, initial_state=initial_state, do=do, dht=dht, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
        return dq, dk, dv, dg, None, dh0, None, None, None


def chunk_gla(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', scale: 'Optional[int]'=None, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]` applied to keys.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gla import chunk_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = chunk_gla(q, k, v, g,
                              initial_state=h0,
                              output_final_state=True,
                              head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gla(q, k, v, g,
                                      initial_state=h0,
                                      output_final_state=True,
                                      cu_seqlens=cu_seqlens,
                                      head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = ChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state, cu_seqlens, head_first)
    return o, final_state


def ceildiv(a, b):
    return -(a // -b)


def pad(x, chunk_size=16):
    T = x.shape[-2]
    padded_seq_len = ceildiv(T, chunk_size) * chunk_size
    if x.shape[-2] % chunk_size != 0:
        x = F.pad(x, (0, 0, 0, padded_seq_len - T))
    return x


def fused_chunk_gla(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', scale: 'int'=-1, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    if scale == -1:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v, g = map(lambda x: x.transpose(1, 2), (q, k, v, g))
    seq_len = q.shape[-2]
    q, k, v, g = map(lambda x: pad(x), [q, k, v, g])
    o, final_state = FusedChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state)
    o = o[..., :seq_len, :].contiguous()
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state


def fused_recurrent(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None, gk: 'Optional[torch.Tensor]'=None, gv: 'Optional[torch.Tensor]'=None, scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, reverse: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True):
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return FusedRecurrentFunction.apply(q, k, v, g, gk, gv, scale, initial_state, output_final_state, reverse, cu_seqlens, head_first)


def fused_recurrent_gla(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', gk: 'Optional[torch.Tensor]'=None, gv: 'Optional[torch.Tensor]'=None, scale: 'Optional[int]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, reverse: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        gk (torch.Tensor):
            Forget gates of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]` applied to keys.
        gv (torch.Tensor):
            Forget gates of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]` applied to values.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gla import fused_recurrent_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_gla(q, k, v, g,
                                        initial_state=h0,
                                        output_final_state=True,
                                        head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_gla(q, k, v, g,
                                                initial_state=h0,
                                                output_final_state=True,
                                                cu_seqlens=cu_seqlens,
                                                head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = fused_recurrent(q=q, k=k, v=v, g=None, gk=gk, gv=gv, scale=scale, initial_state=initial_state, output_final_state=output_final_state, reverse=reverse, cu_seqlens=cu_seqlens, head_first=head_first)
    return o, final_state


def chunk_gsa_bwd_k(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', h: 'torch.Tensor', h0: 'torch.Tensor', o: 'torch.Tensor', do: 'torch.Tensor', dht: 'torch.Tensor', dg: 'torch.Tensor', scale: 'float'=1.0, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    HQ = q.shape[1] if head_first else q.shape[2]
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    NG = HQ // H
    if h is None:
        h, _ = chunk_fwd_h(k=k, v=v, g=None, gk=None, gv=g, h0=h0, output_final_state=False, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT, states_in_fp32=False)
    dh, dh0 = chunk_bwd_dh(q=q, k=k, v=v, g=None, gk=None, gv=g, do=do, h0=h0, dht=dht, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT, states_in_fp32=True)
    dA = q.new_empty(NV, B, *((HQ, T) if head_first else (T, HQ)), BT)
    grid = NV, NT * NC * NC, B * HQ
    chunk_gsa_bwd_k_kernel_dA[grid](v, g, do, dA, offsets=offsets, indices=indices, scale=scale, B=B, T=T, HQ=HQ, H=H, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG, HEAD_FIRST=head_first)
    dA = dA.sum(0, dtype=dA.dtype)
    A = do.new_empty(NK, B, *((HQ, T) if head_first else (T, HQ)), BT)
    dq = torch.empty_like(q)
    dk = k.new_empty(B, *((HQ, T) if head_first else (T, HQ)), K)
    dv = v.new_empty(NK, B, *((HQ, T) if head_first else (T, HQ)), V)
    dgv = g.new_empty(NK, B, *((HQ, T) if head_first else (T, HQ)), V, dtype=torch.float)
    grid = NK, NT, B * HQ
    chunk_gsa_bwd_k_kernel_dqkvg[grid](q, k, v, h, g, A, do, dh, dq, dk, dv, dg, dgv, dA, offsets=offsets, indices=indices, scale=scale, B=B, T=T, HQ=HQ, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, NG=NG, HEAD_FIRST=head_first)
    A = A.sum(0, dtype=A.dtype)
    dv = dv.sum(0, dtype=dv.dtype)
    dgv = dgv.sum(0, dtype=dgv.dtype)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT * NC, B * HQ
    chunk_gsa_bwd_k_kernel_intra_dvg[grid](v, g, o, A, do, dv, dg, offsets=offsets, indices=indices, T=T, HQ=HQ, H=H, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG, HEAD_FIRST=head_first, num_warps=4, num_stages=2)
    dg = dgv.add_(chunk_local_cumsum(dg, chunk_size=BT, reverse=True, offsets=offsets, indices=indices, head_first=head_first))
    return dq, dk, dv, dg, dh0


def chunk_gsa_bwd_v(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', h0: 'torch.Tensor', h: 'torch.Tensor', A: 'torch.Tensor', do: 'torch.Tensor', dht: 'torch.Tensor', dg: 'torch.Tensor', scale: 'float'=1.0, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    dq, dk, dv, dg, dh0 = chunk_gla_bwd(q=q, k=k, v=v, g=None, g_cumsum=g, scale=scale, initial_state=h0, h=h, A=A, do=do, dht=dht, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    return dq, dk, dv, dg, dh0


def softmax_bwd(p: 'torch.Tensor', dp: 'torch.Tensor', dtype: 'Optional[torch.dtype]'=torch.float) ->torch.Tensor:
    shape = p.shape
    p = p.view(-1, p.shape[-1])
    ds = torch.empty_like(p, dtype=dtype)
    N, D = p.shape
    B = triton.next_power_of_2(D)
    softmax_bwd_kernel[N,](p=p, dp=dp, ds=ds, D=D, B=B)
    return ds.view(*shape)


def chunk_gsa_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', g: 'torch.Tensor', ok: 'torch.Tensor', p: 'torch.Tensor', A: 'Tuple[torch.Tensor, torch.Tensor]', h: 'Tuple[torch.Tensor, torch.Tensor]', initial_state: 'Optional[Tuple[torch.Tensor, torch.Tensor]]', scale: 'float', do: 'torch.Tensor', dht: 'Tuple[torch.Tensor, torch.Tensor]', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    _, Av = A
    hk, hv = h
    dhkt, dhvt = dht
    qv = p
    dqv, dsv, dv, dg, dhv0 = chunk_gsa_bwd_v(q=qv, k=s, v=v, g=g, h0=hv0, h=hv, A=Av, do=do, dht=dhvt, dg=None, scale=1.0, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    dok = softmax_bwd(p, dqv, dtype=ok.dtype)
    dq, dk, dsk, dg, dhk0 = chunk_gsa_bwd_k(q=q, k=k, v=s, g=g, h0=hk0, h=hk, o=ok, do=dok, dht=dhkt, dg=dg, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    ds = dsv.add_(dsk)
    if q.shape[1] != k.shape[1]:
        dk, dv, ds, dg = map(lambda x: reduce(x, 'b (h g) ... -> b h ...', 'sum', h=k.shape[1]), (dk, dv, ds, dg))
    dg = dg
    return dq, dk, dv, ds, dg, dhk0, dhv0


def chunk_gsa_fwd_k(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', h0: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, scale: 'float'=1.0, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BV = min(64, triton.next_power_of_2(V))
    HQ = q.shape[1] if head_first else q.shape[2]
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NC = triton.cdiv(BT, BC)
    NG = HQ // H
    h, ht = chunk_fwd_h(k=k, v=v, g=None, gk=None, gv=g, h0=h0, output_final_state=output_final_state, offsets=offsets, head_first=head_first, chunk_size=BT, states_in_fp32=False)
    o = v.new_empty(B, *((HQ, T) if head_first else (T, HQ)), V)
    A = q.new_empty(B, *((HQ, T) if head_first else (T, HQ)), BT)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * HQ
    chunk_gsa_fwd_k_kernel_inter[grid](q, k, h, g, o, A, offsets=offsets, indices=indices, scale=scale, T=T, HQ=HQ, H=H, K=K, V=V, BT=BT, NG=NG, HEAD_FIRST=head_first)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT * NC, B * HQ
    chunk_gsa_fwd_k_kernel_intra[grid](v, g, o, A, offsets=offsets, indices=indices, T=T, HQ=HQ, H=H, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG, HEAD_FIRST=head_first, num_warps=4, num_stages=2)
    return A, h, ht, o


def chunk_gsa_fwd_v(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', scale: 'float'=1.0, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, A, h, ht, o = chunk_gla_fwd(q=q, k=k, v=v, g=None, g_cumsum=g, scale=scale, initial_state=initial_state, output_final_state=output_final_state, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    return A, h, ht, o


def softmax_fwd(x: 'torch.Tensor', dtype: 'Optional[torch.dtype]'=torch.float) ->torch.Tensor:
    shape = x.shape
    x = x.view(-1, x.shape[-1])
    N, D = x.shape
    B = triton.next_power_of_2(D)
    p = torch.empty_like(x, dtype=dtype)
    softmax_fwd_kernel[N,](x=x, p=p, D=D, B=B)
    return p.view(*shape)


def chunk_gsa_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', g: 'torch.Tensor', initial_state: 'Optional[Tuple[torch.Tensor, torch.Tensor]]'=None, output_final_state: 'bool'=False, scale: 'float'=1.0, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    Ak, hk, hkt, ok = chunk_gsa_fwd_k(q=q, k=k, v=s, g=g, h0=hk0, output_final_state=output_final_state, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    p = softmax_fwd(ok, dtype=torch.float)
    qv = p
    Av, hv, hvt, ov = chunk_gsa_fwd_v(q=qv, k=s, v=v, g=g, scale=1.0, initial_state=hv0, output_final_state=output_final_state, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    return Ak, hk, hkt, ok, p, Av, hv, hvt, ov


class ChunkGSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', g: 'torch.Tensor', scale: 'float', hk0: 'Optional[torch.Tensor]', hv0: 'Optional[torch.Tensor]', output_final_state: 'bool', checkpoint_level: 'int', offsets: 'Optional[torch.LongTensor]', head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        g_org, g = g, chunk_local_cumsum(g, chunk_size, offsets=offsets, indices=indices, head_first=head_first)
        Ak, hk, hkt, ok, p, Av, hv, hvt, ov = chunk_gsa_fwd(q=q, k=k, v=v, s=s, g=g, initial_state=(hk0, hv0), output_final_state=output_final_state, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
        if checkpoint_level >= 1:
            del g
            g = g_org
        if checkpoint_level > 1:
            del hk
            del hv
            hk, hv = None, None
        else:
            hk0, hv0 = None, None
        ctx.save_for_backward(q, k, v, s, g, ok, p, Av, hk0, hv0, hk, hv)
        ctx.checkpoint_level = checkpoint_level
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        ctx.chunk_size = chunk_size
        return ov, hkt, hvt

    @staticmethod
    @contiguous
    def backward(ctx, dov, dhkt=None, dhvt=None):
        q, k, v, s, g, ok, p, Av, hk0, hv0, hk, hv = ctx.saved_tensors
        scale = ctx.scale
        offsets = ctx.offsets
        indices = ctx.indices
        head_first = ctx.head_first
        chunk_size = ctx.chunk_size
        if ctx.checkpoint_level >= 1:
            g = chunk_local_cumsum(g, chunk_size, offsets=offsets, indices=indices, head_first=head_first)
        dq, dk, dv, ds, dg, dhk0, dhv0 = chunk_gsa_bwd(q=q, k=k, v=v, s=s, g=g, ok=ok, p=p, A=(None, Av), h=(hk, hv), initial_state=(hk0, hv0), scale=scale, do=dov, dht=(dhkt, dhvt), offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
        return dq, dk, dv, ds, dg, None, dhk0, dhv0, None, None, None, None


def chunk_gsa(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None, scale: 'Optional[int]'=None, initial_state: 'Optional[Tuple[torch.Tensor]]'=None, output_final_state: 'Optional[bool]'=False, checkpoint_level: 'Optional[int]'=2, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'Optional[bool]'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, HQ, T, K]` if `head_first=True` else `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
            GQA is performed if `H` is not equal to `HQ`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, H, T, M]` if `head_first=True` else `[B, T, H, M]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, M]` applied to keys.
            If not provided, this function is equivalent to vanilla ABC.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state tuple, having tensors of shape `[N, H, K, M]` and `[N, H, M, V]`.
            Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `2`:
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the fp32 cumulative values during backward.
            - Level `2`: recompute the fp32 cumulative values and forward hidden states during backward.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` if `output_final_state=True`.
            `None` otherwise.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gsa import fused_recurrent_gsa
        # inputs with equal lengths
        >>> B, T, H, K, V, M = 4, 2048, 4, 512, 512, 64
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> s = torch.randn(B, T, H, M, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, M, device='cuda'))
        >>> h0 = (torch.randn(B, H, K, M, device='cuda'), torch.randn(B, H, M, V, device='cuda'))
        >>> o, (hk, hv) = chunk_gsa(q, k, v, s, g,
                                    initial_state=h0,
                                    output_final_state=True,
                                    head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, s, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, s, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, (hk_var, hv_var) = chunk_gsa(q, k, v, s, g,
                                                initial_state=h0,
                                                output_final_state=True,
                                                cu_seqlens=cu_seqlens,
                                                head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert hk.allclose(hk_var)
        >>> assert hv.allclose(hv_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}.')
    assert checkpoint_level in [0, 1, 2]
    if g is None:
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s = torch.exp(s - z)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    o, *final_state = ChunkGSAFunction.apply(q, k, v, s, g, scale, hk0, hv0, output_final_state, checkpoint_level, cu_seqlens, head_first)
    return o, final_state


@contiguous
def chunk_global_cumsum_scalar(s: 'torch.Tensor', dtype: 'Optional[torch.dtype]'=None, reverse: 'bool'=False, offsets: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True, output_dtype: 'Optional[torch.dtype]'=torch.float) ->torch.Tensor:
    dtype = dtype or s.dtype
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    if offsets is not None:
        B = len(offsets) - 1
    grid = B * H,
    z = torch.empty_like(s, dtype=output_dtype or dtype)
    chunk_global_cumsum_scalar_kernel[grid](s, z, offsets, T=T, H=H, HEAD_FIRST=head_first, REVERSE=reverse)
    return z


@contiguous
def chunk_global_cumsum_vector(s: 'torch.Tensor', dtype: 'Optional[torch.dtype]'=None, reverse: 'bool'=False, offsets: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True, output_dtype: 'Optional[torch.dtype]'=torch.float) ->torch.Tensor:
    dtype = dtype or s.dtype
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    BS = min(32, triton.next_power_of_2(S))
    if offsets is not None:
        B = len(offsets) - 1
    grid = triton.cdiv(S, BS), B * H
    z = torch.empty_like(s, dtype=output_dtype or dtype)
    chunk_global_cumsum_vector_kernel[grid](s, z, offsets, T=T, H=H, S=S, BS=BS, HEAD_FIRST=head_first, REVERSE=reverse)
    return z


@contiguous
def chunk_global_cumsum(s: 'torch.Tensor', dtype: 'Optional[torch.dtype]'=None, reverse: 'bool'=False, offsets: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True, output_dtype: 'Optional[torch.dtype]'=torch.float) ->torch.Tensor:
    if offsets is not None:
        assert s.shape[0] == 1, 'Only batch size 1 is supported when offsets are provided'
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(s, dtype, reverse, offsets, head_first, output_dtype)
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(s, dtype, reverse, offsets, head_first, output_dtype)
    else:
        raise ValueError(f'Unsupported input shape {s.shape}. which should be [B, H, T]/[B, H, T, D] if `head_first=True` or [B, T, H]/[B, T, H, D] otherwise')


def fused_recurrent_gsa_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', g: 'torch.Tensor', qv: 'torch.Tensor', hk0: 'Optional[torch.Tensor]'=None, hv0: 'Optional[torch.Tensor]'=None, ok: 'Optional[torch.Tensor]'=None, do: 'Optional[torch.Tensor]'=None, dhkt: 'Optional[torch.Tensor]'=None, dhvt: 'Optional[torch.Tensor]'=None, scale: 'float'=1.0, reverse: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor]:
    if head_first:
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV, BM = min(K, 64), min(V, 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)
    if head_first:
        dqv = q.new_empty(NV, B, H, T, M, dtype=torch.float)
        dsv = q.new_empty(NV, B, H, T, M, dtype=torch.float)
        dv = q.new_empty(NM, B, H, T, V, dtype=torch.float)
    else:
        dqv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
        dsv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
        dv = q.new_empty(NM, B, T, H, V, dtype=torch.float)
    dhk0 = torch.empty_like(hk0) if hk0 is not None else None
    dhv0 = torch.empty_like(hv0) if hv0 is not None else None
    gk, gv = g, None
    grid = NV, NM, N * H
    fused_recurrent_bwd_kernel[grid](q=qv, k=s, v=v, g=None, gk=gk, gv=gv, h0=hv0, do=do, dq=dqv, dk=dsv, dv=dv, dht=dhvt, dh0=dhv0, offsets=offsets, scale=1.0, B=B, T=T, H=H, K=M, V=V, BK=BM, BV=BV, USE_G=False, USE_GK=True, USE_GV=False, REVERSE=reverse, HEAD_FIRST=head_first)
    dqv = dqv.sum(0)
    dsv = dsv.sum(0)
    dv = dv.sum(0)
    dgk = chunk_global_cumsum(dqv * qv.float() - dsv * s.float(), reverse=not reverse, offsets=offsets, head_first=head_first)
    dok = qv * (dqv - (qv * dqv).sum(-1, True))
    if head_first:
        dq = q.new_empty(NM, B, H, T, K, dtype=torch.float)
        dk = q.new_empty(NM, B, H, T, K, dtype=torch.float)
        dsk = q.new_empty(NK, B, H, T, M, dtype=torch.float)
    else:
        dq = q.new_empty(NM, B, T, H, K, dtype=torch.float)
        dk = q.new_empty(NM, B, T, H, K, dtype=torch.float)
        dsk = q.new_empty(NK, B, T, H, M, dtype=torch.float)
    gk, gv = None, g
    grid = NM, NK, N * H
    fused_recurrent_bwd_kernel[grid](q=q, k=k, v=s, g=None, gk=gk, gv=gv, h0=hk0, do=dok, dq=dq, dk=dk, dv=dsk, dht=dhkt, dh0=dhk0, offsets=offsets, scale=scale, B=B, T=T, H=H, K=K, V=M, BK=BK, BV=BM, USE_G=False, USE_GK=False, USE_GV=True, REVERSE=reverse, HEAD_FIRST=head_first)
    dq = dq.sum(0)
    dk = dk.sum(0)
    dsk = dsk.sum(0)
    dgv = chunk_global_cumsum(dok.float() * ok.float() - dsk * s.float(), reverse=not reverse, offsets=offsets, head_first=head_first)
    ds = dsk.add_(dsv)
    dg = dgk.add_(dgv)
    return dq, dk, dv, ds, dg, dhk0, dhv0


def fused_recurrent_gsa_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', g: 'torch.Tensor', initial_state: 'Optional[Tuple[torch.Tensor, torch.Tensor]]'=None, output_final_state: 'bool'=False, scale: 'float'=1.0, reverse: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    if head_first:
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    HQ = q.shape[1] if head_first else q.shape[2]
    if HQ != H:
        raise ValueError('GQA not supported yet.')
    BK, BV, BM = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64), min(M, 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        hkt, hvt = q.new_empty(N, H, K, M, dtype=torch.float), q.new_empty(N, H, M, V, dtype=torch.float)
    ok = q.new_empty(NK, *s.shape, dtype=torch.float)
    gk, gv = None, g
    grid = NM, NK, N * H
    fused_recurrent_fwd_kernel[grid](q=q, k=k, v=s, g=None, gk=gk, gv=gv, o=ok, h0=hk0, ht=hkt, offsets=offsets, scale=scale, B=B, T=T, H=H, K=K, V=M, BK=BK, BV=BM, USE_G=False, USE_GK=False, USE_GV=True, REVERSE=reverse, HEAD_FIRST=head_first)
    ok = ok.sum(0)
    qv = ok.softmax(-1, dtype=torch.float)
    ov = q.new_empty(NM, *v.shape, dtype=torch.float)
    gk, gv = g, None
    grid = NV, NM, N * H
    fused_recurrent_fwd_kernel[grid](q=qv, k=s, v=v, g=None, gk=gk, gv=gv, o=ov, h0=hv0, ht=hvt, offsets=offsets, scale=1.0, B=B, T=T, H=H, K=M, V=V, BK=BM, BV=BV, USE_G=False, USE_GK=True, USE_GV=False, REVERSE=reverse, HEAD_FIRST=head_first)
    ov = ov.sum(0)
    return ok, hkt, qv, ov, hvt


def fused_recurrent_gsa_inference(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', g: 'torch.Tensor', initial_state: 'Optional[Tuple[torch.Tensor, torch.Tensor]]'=None, output_final_state: 'bool'=False, scale: 'float'=1.0, head_first: 'bool'=True) ->torch.Tensor:
    if head_first:
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    else:
        B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    HQ = q.shape[1] if head_first else q.shape[2]
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NG = HQ // H
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        if NG == 1:
            hkt, hvt = hk0, hv0
        else:
            hkt, hvt = q.new_empty(B, H, K, M, dtype=torch.float), q.new_empty(B, H, M, V, dtype=torch.float)
    o = v.new_empty(B, HQ, T, V) if head_first else v.new_empty(B, T, HQ, V)
    grid = B * HQ,
    fused_recurrent_gsa_inference_kernel[grid](q, k, v, s, g, o, hk0, hv0, hkt, hvt, scale=scale, K=K, V=V, M=M, BK=BK, BV=BV, NG=NG)
    return o, (hkt, hvt)


def fused_recurrent_gsa(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', s: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None, scale: 'Optional[int]'=None, initial_state: 'Optional[Tuple[torch.Tensor]]'=None, output_final_state: 'Optional[bool]'=False, reverse: 'Optional[bool]'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, H, T, M]` if `head_first=True` else `[B, T, H, M]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, M]` applied to keys.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]` and `[N, H, M, V]`.
            Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gsa import fused_recurrent_gsa
        # inputs with equal lengths
        >>> B, T, H, K, V, M = 4, 2048, 4, 512, 512, 64
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> s = torch.randn(B, T, H, M, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, M, device='cuda'))
        >>> h0 = (torch.randn(B, H, K, M, device='cuda'), torch.randn(B, H, M, V, device='cuda'))
        >>> o, (hk, hv) = fused_recurrent_gsa(q, k, v, s, g,
                                              initial_state=h0,
                                              output_final_state=True,
                                              head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, s, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, s, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, (hk_var, hv_var) = fused_recurrent_gsa(q, k, v, s, g,
                                                          initial_state=h0,
                                                          output_final_state=True,
                                                          cu_seqlens=cu_seqlens,
                                                          head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert hk.allclose(hk_var)
        >>> assert hv.allclose(hv_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}.')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if initial_state is None:
        initial_state = None, None
    o, *final_state = FusedRecurrentGSAFunction.apply(q, k, v, s, g, scale, *initial_state, output_final_state, reverse, cu_seqlens, head_first)
    return o, final_state


class LayerNormLinearFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, norm_weight, norm_bias, linear_weight, linear_bias, residual=None, eps=1e-05, prenorm=False, residual_in_fp32=False, is_rms_norm=False, num_groups=1):
        x_shape_og = x.shape
        if x.shape[-1] % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        x = x.reshape(-1, x.shape[-1] // num_groups)
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape_as(x)
        residual_dtype = residual.dtype if residual is not None else torch.float32 if residual_in_fp32 else None
        y, mean, rstd, residual_out = layer_norm_fwd(x, norm_weight, norm_bias, eps, residual, out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(), residual_dtype=residual_dtype, is_rms_norm=is_rms_norm, num_groups=num_groups)
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight
        linear_bias = linear_bias if linear_bias is not None else None
        out = F.linear(y, linear_weight, linear_bias)
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.num_groups = num_groups
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dout, *args):
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dy = dy.reshape(-1, dy.shape[-1] // ctx.num_groups)
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, x.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(dy, x, norm_weight, norm_bias, ctx.eps, mean, rstd, dresidual, ctx.has_residual, ctx.is_rms_norm, x_dtype=ctx.x_dtype, recompute_output=True, num_groups=ctx.num_groups)
        dlinear_weight = torch.einsum('bo,bi->oi', dout, y.view(-1, linear_weight.shape[-1]))
        return dx.reshape(ctx.x_shape_og), dnorm_weight, dnorm_bias, dlinear_weight, dlinear_bias, dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None, None, None, None, None, None


def layer_norm_linear(x: 'torch.Tensor', norm_weight: 'torch.Tensor', norm_bias: 'torch.Tensor', linear_weight: 'torch.Tensor', linear_bias: 'torch.Tensor', residual: 'torch.Tensor'=None, eps: 'float'=1e-05, prenorm: 'bool'=False, residual_in_fp32: 'bool'=False, is_rms_norm: 'bool'=False, num_groups: 'int'=1):
    return LayerNormLinearFunction.apply(x, norm_weight, norm_bias, linear_weight, linear_bias, residual, eps, prenorm, residual_in_fp32, is_rms_norm, num_groups)


def rms_norm_linear(x: 'torch.Tensor', norm_weight: 'torch.Tensor', norm_bias: 'torch.Tensor', linear_weight: 'torch.Tensor', linear_bias: 'torch.Tensor', residual: 'torch.Tensor'=None, eps: 'float'=1e-05, prenorm: 'bool'=False, residual_in_fp32: 'bool'=False):
    return layer_norm_linear(x=x, norm_weight=norm_weight, norm_bias=norm_bias, linear_weight=linear_weight, linear_bias=linear_bias, residual=residual, eps=eps, prenorm=prenorm, residual_in_fp32=residual_in_fp32, is_rms_norm=True)


class ChunkHGRNFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, T, D = x.shape
        BT, BD = 128, min(64, triton.next_power_of_2(D))
        num_warps = 8 if BD == 64 else 4
        gc = torch.empty_like(g, dtype=torch.float)
        o = torch.empty_like(x, dtype=torch.float)

        def grid(meta):
            return triton.cdiv(D, meta['BD']), triton.cdiv(T, meta['BT']), B
        chunk_hgrn_fwd_kernel_h[grid](x, g, gc, o, initial_state, T=T, D=D, BT=BT, USE_INITIAL_STATE=initial_state is not None)

        def grid(meta):
            return triton.cdiv(D, meta['BD']), B
        chunk_hgrn_fwd_kernel_o[grid](gc, o, o.stride(-3), o.stride(-2), o.stride(-1), T=T, D=D, BT=BT, BD=BD, num_warps=num_warps)
        final_state = None
        if output_final_state:
            final_state = o[:, -1].clone()
        o = o
        ctx.save_for_backward(g, o, initial_state)
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        B, T, D = do.shape
        BT, BD = 128, min(64, triton.next_power_of_2(D))
        num_warps = 8 if BD == 64 else 4
        gc = torch.empty_like(g, dtype=torch.float)
        dx = torch.empty_like(o, dtype=torch.float)

        def grid(meta):
            return triton.cdiv(D, meta['BD']), triton.cdiv(T, meta['BT']), B
        chunk_hgrn_bwd_kernel_h[grid](g, gc, dx, do, T=T, D=D, BT=BT)
        dg = torch.empty_like(g, dtype=torch.float)

        def grid(meta):
            return triton.cdiv(D, meta['BD']), B
        chunk_hgrn_bwd_kernel_o[grid](g, gc, o, dx, dg, o.stride(-3), o.stride(-2), o.stride(-1), T=T, D=D, BT=BT, BD=BD, num_warps=num_warps)
        if initial_state is not None:
            dg[:, 0] = initial_state * dx[:, 0] * g[:, 0].float().exp()
        return dx, dg, None, None


def chunk_hgrn(x: 'torch.Tensor', g: 'torch.Tensor', initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    return ChunkHGRNFunction.apply(x, g, initial_state, output_final_state)


def fused_recurrent_hgrn_bwd(g: 'torch.Tensor', o: 'torch.Tensor', do: 'torch.Tensor', dht: 'torch.Tensor'=None, initial_state: 'torch.Tensor'=None, offsets: 'Optional[torch.LongTensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    B, T, D = do.shape
    N = B if offsets is None else len(offsets) - 1
    dx = torch.empty_like(o, dtype=torch.float)
    dg = torch.empty_like(g, dtype=torch.float)
    dh0 = torch.empty_like(initial_state, dtype=torch.float) if initial_state is not None else None

    def grid(meta):
        return triton.cdiv(D, meta['BD']), N
    fused_recurrent_hgrn_bwd_kernel[grid](g=g, o=o, h0=initial_state, dx=dx, dg=dg, do=do, dht=dht, dh0=dh0, offsets=offsets, T=T, D=D)
    return dx, dg, dh0


def fused_recurrent_hgrn_fwd(x: 'torch.Tensor', g: 'torch.Tensor', initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    N = B if offsets is None else len(offsets) - 1
    o = torch.empty_like(x)
    final_state = x.new_empty(N, D) if output_final_state else None

    def grid(meta):
        return triton.cdiv(D, meta['BD']), N
    fused_recurrent_hgrn_fwd_kernel[grid](x=x, g=g, o=o, h0=initial_state, ht=final_state, offsets=offsets, T=T, D=D)
    return o, final_state


class FusedRecurrentHGRNFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x: 'torch.Tensor', g: 'torch.Tensor', initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None):
        o, ht = fused_recurrent_hgrn_fwd(x=x, g=g, initial_state=initial_state, output_final_state=output_final_state, offsets=offsets)
        ctx.save_for_backward(g, o, initial_state)
        ctx.offsets = offsets
        return o, ht

    @staticmethod
    @contiguous
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        offsets = ctx.offsets
        dx, dg, dh0 = fused_recurrent_hgrn_bwd(g=g, o=o, do=do, dht=dht, initial_state=initial_state, offsets=offsets)
        return dx, dg, dh0, None, None


def fused_recurrent_hgrn(x: 'torch.Tensor', g: 'torch.Tensor', initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (torch.Tensor):
            inputs of shape `[B, T, D].
        g (torch.Tensor):
            Forget gates of shape `[B, T, D]`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, D]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, D]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, D]`.
        final_state (torch.Tensor):
            Final state of shape `[N, D]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.hgrn import fused_recurrent_hgrn
        # inputs with equal lengths
        >>> B, T, D = 4, 2048, 512
        >>> x = torch.randn(B, T, D, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, D, device='cuda'))
        >>> h0 = torch.randn(B, D, device='cuda')
        >>> o, ht = fused_recurrent_hgrn(x, g, initial_state=h0, output_final_state=True)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> x, g = map(lambda x: rearrange(x, 'b t d -> 1 (b t) d'), (x, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = x.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_hgrn(x, g, initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    return FusedRecurrentHGRNFunction.apply(x, g, initial_state, output_final_state, cu_seqlens)


class LayerNormSwishGateLinearFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, o, norm_weight, norm_bias, linear_weight, linear_bias, residual=None, eps=1e-06, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
        x_shape_og = x.shape
        o_shape_og = o.shape
        x = x.reshape(-1, x.shape[-1])
        o = o.reshape(-1, o.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = residual.dtype if residual is not None else torch.float if residual_in_fp32 else None
        y, mean, rstd, residual_out = layer_norm_fwd(x, o, norm_weight, norm_bias, eps, residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm)
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight
        linear_bias = linear_bias if linear_bias is not None else None
        out = F.linear(y, linear_weight, linear_bias)
        ctx.save_for_backward(residual_out, o, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.o_shape_og = o_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dout, *args):
        x, o, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, do, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(dy, x, o, norm_weight, norm_bias, ctx.eps, mean, rstd, dresidual=dresidual, has_residual=ctx.has_residual, is_rms_norm=ctx.is_rms_norm, x_dtype=ctx.x_dtype, recompute_output=True)
        dlinear_weight = torch.einsum('bo,bi->oi', dout, y)
        return dx.reshape(ctx.x_shape_og), do.reshape(ctx.o_shape_og), dnorm_weight, dnorm_bias, dlinear_weight, dlinear_bias, dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None, None, None, None, None


def rms_norm_swish_gate_linear_fn(x, o, norm_weight, norm_bias, linear_weight, linear_bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-06):
    return LayerNormSwishGateLinearFn.apply(x, o, norm_weight, norm_bias, linear_weight, linear_bias, residual, eps, prenorm, residual_in_fp32, True)


class FusedRecurrentLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, scale, initial_state=None, output_final_state=False):
        B, H, T, K = q.shape
        V = v.shape[-1]
        BK, BV = min(K, 32), min(V, 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_warps = 1
        num_stages = 1
        o = q.new_empty(NK, B, H, T, V)
        final_state = q.new_empty(B, H, K, V) if output_final_state else None
        grid = NV, NK, B * H
        fused_recurrent_linear_attn_fwd_kernel[grid](q, k, v, o, initial_state, final_state, q.stride(1), v.stride(1), scale, B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV, USE_INITIAL_STATE=initial_state is not None, STORE_FINAL_STATE=final_state is not None, num_warps=num_warps, num_stages=num_stages)
        o = o.sum(0)
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = ctx.scale
        BK, BV = min(K, 32), min(V, 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_warps = 1
        num_stages = 1
        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        grid = NV, NK, B * H
        fused_recurrent_linear_attn_bwd_kernel[grid](q, k, v, do, dq, dk, dv, initial_state, q.stride(1), v.stride(1), scale, B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV, USE_INITIAL_STATE=initial_state is not None, num_warps=num_warps, num_stages=num_stages)
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq, dk, dv, None, None, None


def fused_recurrent_linear_attn(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, normalize: 'bool'=False, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, final_state = FusedRecurrentLinearAttentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    if normalize:
        o = normalize_output(q * scale, k, o)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state


class LinearAttention(nn.Module):

    def __init__(self, mode: 'str'='chunk', hidden_size: 'str'=1024, expand_k: 'int'=1.0, expand_v: 'int'=1.0, num_heads: 'int'=8, num_kv_heads: 'Optional[int]'=None, feature_map: 'str'='elementwise_product', tie_feature_map_qk: 'bool'=False, output_norm: 'str'='rmsnorm', norm_q: 'bool'=False, norm_k: 'bool'=False, do_feature_map_norm: 'bool'=False, elementwise_affine: 'bool'=True, norm_eps: 'float'=1e-05, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f'Not suppoerted mode `{mode}`.'
        assert self.key_dim % num_heads == 0, f'key dim must be divisible by num_heads of {num_heads}'
        assert self.value_dim % num_heads == 0, f'value dim must be divisible by num_heads of {num_heads}'
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.do_feature_map_norm = do_feature_map_norm
        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)
        elif feature_map == 'elu':

            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        else:
            raise NotImplementedError(f'Not supported feature map `{feature_map}`.')
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        if output_norm == 'rmsnorm':
            self.norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        elif output_norm == 'identity':
            self.norm = nn.Identity()
        else:
            raise NotImplementedError(f'Not supported output norm `{output_norm}`.')
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: 'nn.Module'):
        if getattr(module, '_is_hf_initialized', False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, x):
        mode = self.mode
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v = (repeat(x, '... (h d) -> ... (h g) d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v))
        else:
            k, v = (rearrange(x, '... (h d) -> ... h d', h=self.num_kv_heads) for x in (k, v))
        q = self.feature_map_q(q)
        k = self.feature_map_k(k)
        if self.norm_q:
            q = q / (q.sum(-1, True) + 0.0001)
        if self.norm_k:
            k = k / (k.sum(-1, True) + 0.0001)
        if mode == 'chunk':
            o, final_state = chunk_linear_attn(q=q, k=k, v=v, normalize=self.do_feature_map_norm, head_first=False)
        elif mode == 'fused_chunk':
            o, final_state = fused_chunk_linear_attn(q=q, k=k, v=v, normalize=self.do_feature_map_norm)
        elif mode == 'fused_recurrent':
            o, final_state = fused_recurrent_linear_attn(q=q, k=k, v=v, normalize=self.do_feature_map_norm)
        else:
            raise NotImplementedError
        o = self.norm(o)
        o = self.o_proj(o)
        return o


def chunk_retention(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    """
    if head_first:
        n_heads = q.shape[1]
    else:
        n_heads = q.shape[2]
    s = (1 - q.new_tensor(2.0, dtype=torch.float).pow(-5.0 - q.new_tensor(range(n_heads), dtype=torch.float))).log()
    if head_first:
        g = s[None, :, None].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    else:
        g = s[None, None, :].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    return chunk_simple_gla(q=q, k=k, v=v, scale=scale, g=g, initial_state=initial_state, output_final_state=output_final_state, head_first=head_first, cu_seqlens=cu_seqlens)


def fused_chunk_retention(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    o, final_state = FusedChunkRetentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state


def fused_recurrent_simple_gla(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, reverse: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.simple_gla import fused_recurrent_simple_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_simple_gla(q, k, v, g,
                                               initial_state=h0,
                                               output_final_state=True,
                                               head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_simple_gla(q, k, v, g,
                                                       initial_state=h0,
                                                       output_final_state=True,
                                                       cu_seqlens=cu_seqlens,
                                                       head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = fused_recurrent(q=q, k=k, v=v, g=g, scale=scale, initial_state=initial_state, output_final_state=output_final_state, reverse=reverse, cu_seqlens=cu_seqlens, head_first=head_first)
    return o, final_state


def fused_recurrent_retention(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, reverse: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        n_heads = q.shape[1]
    else:
        n_heads = q.shape[2]
    s = (1 - q.new_tensor(2.0, dtype=torch.float).pow(-5.0 - q.new_tensor(range(n_heads), dtype=torch.float))).log()
    if head_first:
        g = s[None, :, None].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    else:
        g = s[None, None, :].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    return fused_recurrent_simple_gla(q=q, k=k, v=v, g=g, scale=scale, initial_state=initial_state, output_final_state=output_final_state, reverse=reverse, cu_seqlens=cu_seqlens, head_first=head_first)


def parallel_simple_gla_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', do: 'torch.Tensor', scale: 'float', chunk_size: 'int'=128, head_first: 'bool'=True, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    BK = min(128, triton.next_power_of_2(k.shape[-1]))
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0
    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(NK, *v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    dg = torch.empty(NK * NV, *g.shape, dtype=torch.float, device=q.device) if g is not None else None
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    grid = NK * NV, NT, B * H
    parallel_simple_gla_bwd_kernel[grid](q=q, k=k, v=v, g=g, do=do, dq=dq, dk=dk, dv=dv, dg=dg, offsets=offsets, indices=indices, scale=scale, B=B, H=H, T=T, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, HEAD_FIRST=head_first)
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg = chunk_global_cumsum(dg.sum(0), reverse=True, head_first=head_first, offsets=offsets) if g is not None else None
    return dq, dk, dv, dg


def parallel_simple_gla_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', scale: 'float', output_attentions: 'bool'=False, chunk_size: 'int'=128, head_first: 'bool'=True, offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    if g is not None:
        g = chunk_local_cumsum(g, chunk_size, offsets=offsets, head_first=head_first)
    grid = NK * NV, NT, B * H
    o = torch.empty(NK, *v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    attn = q.new_zeros(NK, B, H, T, T) if output_attentions else None
    parallel_simple_gla_fwd_kernel[grid](q=q, k=k, v=v, g=g, o=o, attn=attn, scale=scale, offsets=offsets, indices=indices, B=B, H=H, T=T, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, HEAD_FIRST=head_first)
    o = o.sum(0)
    if output_attentions:
        attn = attn.sum(0)
    return o, g, attn


def parallel_simple_gla(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'Optional[torch.Tensor]'=None, scale: 'Optional[float]'=None, output_attentions: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        g (torch.Tensor):
            Forget gates of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, 'batch size must be 1 when cu_seqlens are provided'
        assert not head_first, 'head_first must be False when cu_seqlens are provided'
    if g is not None:
        g = g.float()
    if output_attentions:
        assert cu_seqlens is None, 'output_attentions=True is not supported with variable-length sequences'
    o, attn = ParallelSimpleGLAFunction.apply(q, k, v, g, scale, output_attentions, head_first, cu_seqlens)
    return o, attn


def parallel_retention(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', scale: 'Optional[float]'=None, output_attentions: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`
    """
    if head_first:
        n_heads = q.shape[1]
    else:
        n_heads = q.shape[2]
    s = (1 - q.new_tensor(2.0, dtype=torch.float).pow(-5.0 - q.new_tensor(range(n_heads), dtype=torch.float))).log()
    if head_first:
        g = s[None, :, None].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    else:
        g = s[None, None, :].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    return parallel_simple_gla(q=q, k=k, v=v, scale=scale, g=g, output_attentions=output_attentions, head_first=head_first, cu_seqlens=cu_seqlens)


def layer_norm(x: 'torch.Tensor', weight: 'torch.Tensor', bias: 'torch.Tensor', residual: 'torch.Tensor'=None, eps: 'float'=1e-05, prenorm: 'bool'=False, residual_in_fp32: 'bool'=False, is_rms_norm: 'bool'=False):
    return LayerNormFunction.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, is_rms_norm)


def parallel_rebased(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', eps: 'float'=1e-05, use_scale: 'bool'=True, use_normalize: 'bool'=True, return_both: 'bool'=False, head_first: 'bool'=True):
    assert q.shape[-1] <= 128, 'only support feature dim up to 128'
    if use_scale:
        scale = q.shape[-1] ** -0.5
    else:
        scale = 1
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, z = triton_parallel_based(q, k, v, scale)
    if return_both:
        return o, z
    if use_normalize:
        o = o / (z[..., None] + eps)
    if not head_first:
        o = o.transpose(1, 2)
    return o


class LoRA(nn.Module):

    def __init__(self, input_dim: 'int', output_dim: 'int', low_rank_dim: 'int', bias: 'Optional[bool]'=True, activation: 'Optional[str]'='tanh'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias
        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f'Not supported activation `{activation}`.')
        self.lora = nn.Sequential(nn.Linear(input_dim, low_rank_dim, bias=False), self.activation, nn.Linear(low_rank_dim, output_dim, bias=bias))

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}('
        s += f'input_dim={self.input_dim}, low_rank_dim={self.low_rank_dim}, output_dim={self.output_dim}'
        if not self.bias:
            s += f', bias={self.bias}'
        s += ')'
        return s

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.lora(x)


class DDLerpLinear(nn.Module):

    def __init__(self, input_dim: 'int', output_dim: 'int', low_rank_dim: 'Optional[int]'=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}'
        if self.low_rank_dim is not None:
            s += f', low_rank_dim={self.low_rank_dim}'
        s += ')'
        return s

    def forward(self, x: 'torch.Tensor', mu: 'torch.Tensor', delta: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * mu)


def group_norm(x: 'torch.Tensor', weight: 'torch.Tensor', bias: 'torch.Tensor', residual: 'torch.Tensor'=None, eps: 'float'=1e-05, prenorm: 'bool'=False, residual_in_fp32: 'bool'=False, is_rms_norm: 'bool'=False, num_groups: 'int'=1):
    return LayerNormFunction.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, is_rms_norm, num_groups)


class LerpLinear(nn.Module):

    def __init__(self, input_dim: 'int', output_dim: 'int', low_rank_dim: 'Optional[int]'=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)
        self.mu = nn.Parameter(torch.zeros(input_dim))

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}'
        if self.low_rank_dim is not None:
            s += f', low_rank_dim={self.low_rank_dim}'
        s += ')'
        return s

    def forward(self, x: 'torch.Tensor', delta: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * self.mu)


def chunk_rwkv6_bwd_dh(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', gi: 'torch.Tensor', ge: 'torch.Tensor', do: 'torch.Tensor', h0: 'torch.Tensor', dht: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.Tensor]'=None, indices: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64, states_in_fp32: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(offsets) - 1, len(indices)
        chunk_offsets = torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], BT)]).cumsum(-1)
    NG = HQ // H
    if head_first:
        dh = k.new_empty(B, HQ, NT, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    else:
        dh = k.new_empty(B, NT, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta):
        return triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H
    chunk_rwkv6_bwd_kernel_dh[grid](q=q, gi=gi, ge=ge, do=do, dh=dh, dht=dht, dh0=dh0, offsets=offsets, chunk_offsets=chunk_offsets, scale=scale, T=T, HQ=HQ, H=H, K=K, V=V, BT=BT, NG=NG, HEAD_FIRST=head_first)
    return dh, dh0


def chunk_rwkv6_bwd_dqk_intra(q: 'torch.Tensor', k: 'torch.Tensor', gi: 'torch.Tensor', ge: 'torch.Tensor', dA: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K = q.shape
    else:
        B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    grid = NK, NT * NC, B * H
    chunk_rwkv6_bwd_kernel_intra[grid](q, k, gi, ge, dA, dq, dk, offsets, indices, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC, HEAD_FIRST=head_first)
    return dq, dk


def chunk_rwkv6_bwd_dqkgu(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', h: 'torch.Tensor', g: 'torch.Tensor', gi: 'torch.Tensor', ge: 'torch.Tensor', u: 'torch.Tensor', do: 'torch.Tensor', dh: 'torch.Tensor', dA: 'torch.Tensor', dq: 'torch.Tensor', dk: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    dg = torch.empty_like(g)
    du = u.new_empty(B * NT, H, K, dtype=torch.float)

    def grid(meta):
        return triton.cdiv(K, meta['BK']), NT, B * H
    chunk_rwkv6_bwd_kernel_inter[grid](q, k, v, h, gi, ge, u, do, dh, dA, dq, dk, dq2, dk2, dg, du, offsets, indices, scale, T=T, H=H, K=K, V=V, BT=BT, HEAD_FIRST=head_first)
    du = du.sum(0)
    return dq2, dk2, dg, du


def chunk_rwkv6_fwd_cumsum(g: 'torch.Tensor', chunk_size: 'int', offsets: 'Optional[torch.Tensor]'=None, indices: 'Optional[torch.Tensor]'=None, head_first: 'bool'=True) ->torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    gi, ge = torch.empty_like(g, dtype=torch.float), torch.empty_like(g, dtype=torch.float)

    def grid(meta):
        return triton.cdiv(meta['S'], meta['BS']), NT, B * H
    chunk_rwkv6_fwd_cumsum_kernel[grid](g, gi, ge, offsets, indices, T=T, H=H, S=S, BT=BT, HEAD_FIRST=head_first)
    return gi, ge


def chunk_rwkv6_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', u: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', A: 'torch.Tensor', do: 'torch.Tensor', dht: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    gi, ge = chunk_rwkv6_fwd_cumsum(g, chunk_size=chunk_size, offsets=offsets, indices=indices, head_first=head_first)
    h, _ = chunk_fwd_h(k=k, v=v, g=None, gk=gi, gv=None, h0=initial_state, output_final_state=False, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size, states_in_fp32=True)
    dh, dh0 = chunk_rwkv6_bwd_dh(q=q, k=k, v=v, gi=gi, ge=ge, do=do, h0=initial_state, dht=dht, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size, states_in_fp32=True)
    dA = chunk_gla_bwd_dA(v=v, do=do, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    dv = chunk_gla_bwd_dv(k=k, g=gi, A=A, do=do, dh=dh, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    dq, dk = chunk_rwkv6_bwd_dqk_intra(q=q, k=k, gi=gi, ge=ge, dA=dA, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    dq, dk, dg, du = chunk_rwkv6_bwd_dqkgu(q=q, k=k, v=v, h=h, g=g, gi=gi, ge=ge, u=u, do=do, dh=dh, dA=dA, dq=dq, dk=dk, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    return dq, dk, dv, dg, du, dh0


def chunk_rwkv6_fwd_intra(q: 'torch.Tensor', k: 'torch.Tensor', gi: 'torch.Tensor', ge: 'torch.Tensor', u: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K = k.shape
    else:
        B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    A = q.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=torch.float)
    grid = NT, NC * NC, B * H
    chunk_rwkv6_fwd_A_kernel_intra_sub_inter[grid](q, k, gi, ge, A, offsets, indices, scale, T=T, H=H, K=K, BT=BT, BC=BC, NC=NC, HEAD_FIRST=head_first)
    grid = NT, NC, B * H
    if K <= 256:
        BK = triton.next_power_of_2(K)
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra[grid](q, k, gi, ge, u, A, offsets, indices, scale, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, HEAD_FIRST=head_first)
    else:
        BK = min(128, triton.next_power_of_2(K))
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, *((H, T) if head_first else (T, H)), BC, dtype=torch.float)
        grid = NK, NT * NC, B * H
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split[grid](q, k, gi, ge, u, A_intra, offsets, indices, scale, B=B, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC, HEAD_FIRST=head_first)
        grid = NT, NC, B * H
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge[grid](A_intra, A, offsets, indices, B=B, T=T, H=H, BT=BT, BC=BC, NK=NK, HEAD_FIRST=head_first)
    return A


def chunk_rwkv6_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', u: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gi, ge = chunk_rwkv6_fwd_cumsum(g, chunk_size=chunk_size, offsets=offsets, indices=indices, head_first=head_first)
    h, ht = chunk_fwd_h(k=k, v=v, g=None, gk=gi, gv=None, h0=initial_state, output_final_state=output_final_state, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size, states_in_fp32=False)
    A = chunk_rwkv6_fwd_intra(q=q, k=k, gi=gi, ge=ge, u=u, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    o = chunk_gla_fwd_o_gk(q=q, v=v, g=ge, A=A, h=h, scale=scale, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
    return A, h, ht, o


class ChunkRWKV6Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, g, u, scale, initial_state, output_final_state, offsets, head_first):
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        A, h, ht, o = chunk_rwkv6_fwd(q=q, k=k, v=v, g=g, u=u, scale=scale, initial_state=initial_state, output_final_state=output_final_state, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
        ctx.save_for_backward(q, k, v, g, initial_state, A, u)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        return o, ht

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, g, initial_state, A, u = ctx.saved_tensors
        chunk_size, scale, offsets, indices, head_first = ctx.chunk_size, ctx.scale, ctx.offsets, ctx.indices, ctx.head_first
        dq, dk, dv, dg, du, dh0 = chunk_rwkv6_bwd(q=q, k=k, v=v, g=g, u=u, scale=scale, initial_state=initial_state, A=A, do=do, dht=dht, offsets=offsets, indices=indices, head_first=head_first, chunk_size=chunk_size)
        return dq, dk, dv, dg, du, None, dh0, None, None, None


def chunk_rwkv6(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', g: 'torch.Tensor', u: 'torch.Tensor', scale: 'Optional[int]'=None, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]` applied to keys.
        u (torch.Tensor):
            bonus representations of shape `[H]`.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Optional[torch.Tensor]):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.rwkv6 import chunk_rwkv6
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> u = torch.randn(H, K, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = chunk_rwkv6(q, k, v, g, u,
                                initial_state=h0,
                                output_final_state=True,
                                head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_rwkv6(q, k, v, g, u,
                                        initial_state=h0,
                                        output_final_state=True,
                                        cu_seqlens=cu_seqlens,
                                        head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = ChunkRWKV6Function.apply(q, k, v, g, u, scale, initial_state, output_final_state, cu_seqlens, head_first)
    return o, final_state


def fused_recurrent_rwkv6_bwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', w: 'torch.Tensor', u: 'torch.Tensor', do: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, reverse: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = min(triton.next_power_of_2(K), 16), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    dq = q.new_empty(NV, *q.shape, dtype=torch.float)
    dq1 = torch.empty_like(dq)
    grid = NV, NK, N * H
    fused_recurrent_rwkv6_bwd_kernel_dq[grid](k, v, w, u, do, dq, dq1, initial_state, offsets, scale, B=B, T=T, H=H, K=K, V=V, BK=BK, BV=BV, REVERSE=reverse, HEAD_FIRST=head_first)
    dq = dq.sum(0)
    dq1 = dq1.sum(0)
    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    dk = q.new_empty(NV, *k.shape, dtype=torch.float)
    dk1 = q.new_empty(NV, *k.shape, dtype=torch.float)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float)
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None
    grid = NV, NK, N * H
    fused_recurrent_rwkv6_bwd_kernel_dkv[grid](q, k, v, w, u, do, dk, dk1, dv, dh0, offsets, scale, B=B, T=T, H=H, K=K, V=V, BK=BK, BV=BV, REVERSE=reverse, HEAD_FIRST=head_first)
    dk = dk.sum(0)
    dk1 = dk1.sum(0)
    dv = dv.sum(0)
    dw = torch.empty_like(w)

    def grid(meta):
        return triton.cdiv(meta['K'], meta['BK']), N * H
    fused_recurrent_rwkv6_bwd_kernel_dw[grid](q, k, dq1, dk1, dw, offsets, scale, T=T, H=H, K=K, REVERSE=not reverse, HEAD_FIRST=head_first)
    du = (do.float() * v).sum(-1, True, dtype=torch.float) * q * k * scale
    du = du.sum((0, 2)) if head_first else du.sum((0, 1))
    return dq, dk, dv, dw, du, dh0


def fused_recurrent_rwkv6_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', w: 'torch.Tensor', u: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, reverse: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float)
    grid = NV, NK, N * H
    fused_recurrent_rwkv6_fwd_kernel[grid](q, k, v, w, u, o, h0, ht, offsets, scale, B=B, T=T, H=H, K=K, V=V, BK=BK, BV=BV, REVERSE=reverse, HEAD_FIRST=head_first)
    o = o.sum(0)
    return o, ht


def fused_recurrent_rwkv6(r: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', w: 'torch.Tensor', u: 'torch.Tensor', scale: 'Optional[int]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, reverse: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        r (torch.Tensor):
            reception of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
            Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        w (torch.Tensor):
            data-dependent decays of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]` in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `[H, K]`
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Optional[torch.Tensor]):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.rwkv6 import fused_recurrent_rwkv6
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> u = torch.randn(H, K, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_rwkv6(q, k, v, g, u,
                                          initial_state=h0,
                                          output_final_state=True,
                                          head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_rwkv6(q, k, v, g, u,
                                                  initial_state=h0,
                                                  output_final_state=True,
                                                  cu_seqlens=cu_seqlens,
                                                  head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {r.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = FusedRecurrentRWKV6Function.apply(r, k, v, w, u, scale, initial_state, output_final_state, reverse, cu_seqlens, head_first)
    return o, final_state


def chunk_dplr_bwd_dAu(v: 'torch.Tensor', v_new: 'torch.Tensor', do: 'torch.Tensor', A_qb: 'torch.Tensor', scale: 'float', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->torch.Tensor:
    if head_first:
        B, H, T, V = v.shape
    else:
        B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    BV = min(triton.next_power_of_2(V), 128)
    grid = NT, B * H
    dA_qk = torch.empty(B, H, T, BT, dtype=torch.float, device=v.device) if head_first else torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dA_qb = torch.empty(B, H, T, BT, dtype=torch.float, device=v.device) if head_first else torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dv_new = torch.empty_like(v_new)
    chunk_dplr_bwd_kernel_dAu[grid](v=v, do=do, v_new=v_new, A_qb=A_qb, dA_qk=dA_qk, dA_qb=dA_qb, dv_new=dv_new, offsets=offsets, indices=indices, scale=scale, T=T, H=H, V=V, BT=BT, BV=BV, HEAD_FIRST=head_first)
    return dv_new, dA_qk, dA_qb


def chunk_dplr_bwd_dhu(qg: 'torch.Tensor', bg: 'torch.Tensor', w: 'torch.Tensor', gk: 'torch.Tensor', h0: 'torch.Tensor', dht: 'Optional[torch.Tensor]', do: 'torch.Tensor', dv: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *qg.shape, do.shape[-1]
    else:
        B, T, H, K, V = *qg.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, 'current kernel does not support head dimension being larger than 256.'
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 32
    else:
        BV = 32
        BC = 32
    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    if head_first:
        dh = qg.new_empty(B, H, NT, K, V)
    else:
        dh = qg.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.zeros_like(dv)
    grid = NK, NV, N * H
    chunk_dplr_bwd_kernel_dhu[grid](qg=qg, bg=bg, w=w, gk=gk, dht=dht, dh0=dh0, do=do, dh=dh, dv=dv, dv2=dv2, offsets=offsets, chunk_offsets=chunk_offsets, T=T, H=H, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return dh, dh0, dv2


def chunk_dplr_bwd_dqk_intra(q: 'torch.Tensor', k: 'torch.Tensor', a: 'torch.Tensor', b: 'torch.Tensor', gi: 'torch.Tensor', ge: 'torch.Tensor', dAqk: 'torch.Tensor', dAqb: 'torch.Tensor', dAak: 'torch.Tensor', dAab: 'torch.Tensor', dqg: 'torch.Tensor', dkg: 'torch.Tensor', dag: 'torch.Tensor', dbg: 'torch.Tensor', dgk_last: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, scale: 'float'=1.0, chunk_size: 'int'=64):
    if head_first:
        B, H, T, K = q.shape
    else:
        B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dgk = torch.empty_like(gi, dtype=torch.float)
    dgk_offset = torch.empty_like(gi, dtype=torch.float)
    grid = NK, NT * NC, B * H
    chunk_dplr_bwd_kernel_intra[grid](q=q, k=k, a=a, b=b, gi=gi, ge=ge, dAqk=dAqk, dAqb=dAqb, dAak=dAak, dAab=dAab, dq=dq, dk=dk, dgk=dgk, dgk_offset=dgk_offset, dqg=dqg, dkg=dkg, dag=dag, dbg=dbg, da=da, db=db, offsets=offsets, indices=indices, scale=scale, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC, HEAD_FIRST=head_first)

    def grid2(meta):
        return NT, triton.cdiv(K, meta['BK']), B * H
    dgk_output = torch.empty_like(dgk)
    chunk_dplr_bwd_dgk_kernel[grid2](dgk=dgk, dgk_offset=dgk_offset, dgk_last=dgk_last, dgk_output=dgk_output, offsets=offsets, indices=indices, T=T, H=H, K=K, BT=BT, HEAD_FIRST=head_first)
    return dq, dk, da, db, dgk_output


def chunk_dplr_bwd_dv(A_qk: 'torch.Tensor', kg: 'torch.Tensor', do: 'torch.Tensor', dh: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->torch.Tensor:
    if head_first:
        B, H, T, K, V = *kg.shape, do.shape[-1]
    else:
        B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    dv = torch.empty_like(do)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_dplr_bwd_kernel_dv[grid](A_qk=A_qk, kg=kg, do=do, dv=dv, dh=dh, offsets=offsets, indices=indices, T=T, H=H, K=K, V=V, BT=BT, HEAD_FIRST=head_first)
    return dv


def chunk_dplr_bwd_o(k: 'torch.Tensor', b: 'torch.Tensor', v: 'torch.Tensor', v_new: 'torch.Tensor', gk: 'torch.Tensor', do: 'torch.Tensor', h: 'torch.Tensor', dh: 'torch.Tensor', dv: 'torch.Tensor', w: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, chunk_size: 'int'=64, scale: 'float'=1.0, head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *w.shape, v.shape[-1]
    else:
        B, T, H, K, V = *w.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(k)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)
    grid = NK, NT, B * H
    dA_qk = torch.empty(B, H, T, BT, dtype=torch.float, device=w.device) if head_first else torch.empty(B, T, H, BT, dtype=torch.float, device=w.device)
    dA_qb = torch.empty(B, H, T, BT, dtype=torch.float, device=w.device) if head_first else torch.empty(B, T, H, BT, dtype=torch.float, device=w.device)
    dgk_last = torch.empty(B, H, NT, K, dtype=torch.float, device=w.device) if head_first else torch.empty(B, NT, H, K, dtype=torch.float, device=w.device)
    chunk_dplr_bwd_o_kernel[grid](k=k, b=b, v=v, v_new=v_new, h=h, do=do, dh=dh, dq=dq, dk=dk, db=db, dA_qk=dA_qk, dA_qb=dA_qb, dgk_last=dgk_last, w=w, dv=dv, dw=dw, gk=gk, offsets=offsets, indices=indices, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return dq, dk, dw, db, dgk_last


def chunk_dplr_bwd_wy(A_ab_inv: 'torch.Tensor', A_ak: 'torch.Tensor', v: 'torch.Tensor', ag: 'torch.Tensor', dw: 'torch.Tensor', du: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]', indices: 'Optional[torch.LongTensor]', head_first: 'bool', chunk_size: 'int') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A_ab_inv, A_ak, v, ag, dw, du = map(lambda x: x.contiguous(), [A_ab_inv, A_ak, v, ag, dw, du])
    if head_first:
        B, H, T, K, V = *dw.shape, du.shape[-1]
    else:
        B, T, H, K, V = *dw.shape, du.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    dA_ab = torch.empty_like(A_ab_inv, dtype=torch.float)
    dA_ak = torch.empty_like(A_ak, dtype=torch.float)
    dv = torch.empty_like(v)
    dag = torch.empty_like(ag)
    bwd_prepare_wy_repr_kernel[NT, B * H](A_ab_inv=A_ab_inv, A_ak=A_ak, ag=ag, v=v, dw=dw, du=du, dv=dv, dag=dag, dAak=dA_ak, dAab=dA_ab, offsets=offsets, indices=indices, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, HEAD_FIRST=head_first)
    return dA_ab, dA_ak, dv, dag


def chunk_dplr_fwd_h(kg: 'torch.Tensor', v: 'torch.Tensor', w: 'torch.Tensor', u: 'torch.Tensor', bg: 'torch.Tensor', gk: 'torch.Tensor', initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, offsets: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *kg.shape, u.shape[-1]
    else:
        B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, 'current kernel does not support head dimension larger than 256.'
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64 if K <= 128 else 32
    else:
        BV = 32
        BC = 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    if head_first:
        h = kg.new_empty(B, H, NT, K, V)
    else:
        h = kg.new_empty(B, NT, H, K, V)
    final_state = kg.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)
    grid = NK, NV, N * H
    chunk_dplr_fwd_kernel_h[grid](kg=kg, v=v, w=w, bg=bg, u=u, v_new=v_new, h=h, gk=gk, h0=initial_state, ht=final_state, offsets=offsets, chunk_offsets=chunk_offsets, T=T, H=H, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT, HEAD_FIRST=head_first)
    return h, v_new, final_state


def chunk_dplr_fwd_o(qg: 'torch.Tensor', v: 'torch.Tensor', v_new: 'torch.Tensor', A_qk: 'torch.Tensor', A_qb: 'torch.Tensor', h: 'torch.Tensor', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64) ->torch.Tensor:
    if head_first:
        B, H, T, K, V = *qg.shape, v.shape[-1]
    else:
        B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        NT = len(indices)
    o = torch.empty_like(v)
    grid = lambda meta: (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_dplr_fwd_kernel_o[grid](qg=qg, v=v, v_new=v_new, A_qk=A_qk, A_qb=A_qb, h=h, o=o, offsets=offsets, indices=indices, T=T, H=H, K=K, V=V, BT=BT, HEAD_FIRST=head_first)
    return o


def chunk_fwd_intra_dplr_fn(q: 'torch.Tensor', k: 'torch.Tensor', a: 'torch.Tensor', b: 'torch.Tensor', gi: 'torch.Tensor', ge: 'torch.Tensor', scale: 'float', chunk_size: 'int', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True):
    if head_first:
        B, H, T, K = k.shape
    else:
        B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    Aqk = q.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=q.dtype)
    Aqb = q.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=q.dtype)
    Aab = q.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=torch.float)
    Aak = q.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=torch.float)
    grid = NT, NC * NC, B * H
    chunk_dplr_fwd_A_kernel_intra_sub_inter[grid](q=q, k=k, a=a, b=b, gi=gi, ge=ge, Aqk=Aqk, Aqb=Aqb, Aab=Aab, Aak=Aak, offsets=offsets, indices=indices, scale=scale, T=T, H=H, K=K, BT=BT, BC=BC, NC=NC, HEAD_FIRST=head_first)
    grid = NT, NC, B * H
    BK = triton.next_power_of_2(K)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k, dtype=q.dtype)
    ag = torch.empty_like(a, dtype=q.dtype)
    bg = torch.empty_like(b, dtype=q.dtype)
    chunk_dplr_fwd_A_kernel_intra_sub_intra[grid](q=q, k=k, a=a, b=b, gi=gi, ge=ge, Aqk=Aqk, Aqb=Aqb, Aab=Aab, Aak=Aak, qg=qg, kg=kg, ag=ag, bg=bg, offsets=offsets, indices=indices, scale=scale, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, HEAD_FIRST=head_first, NC=NC)
    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg


def chunk_dplr_fwd(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', a: 'torch.Tensor', b: 'torch.Tensor', gk: 'torch.Tensor', scale: 'float', initial_state: 'torch.Tensor', output_final_state: 'bool', offsets: 'Optional[torch.LongTensor]'=None, indices: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=True, chunk_size: 'int'=64):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT, offsets=offsets, indices=indices, head_first=head_first)
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_fwd_intra_dplr_fn(q=q, k=k, a=a, b=b, gi=gi, ge=ge, scale=scale, offsets=offsets, indices=indices, chunk_size=BT, head_first=head_first)
    w, u, A_ab_inv = fwd_prepare_wy_repr(ag=ag, A_ab=A_ab, A_ak=A_ak, v=v, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    h, v_new, final_state = chunk_dplr_fwd_h(kg=kg, bg=bg, v=v, w=w, u=u, gk=gi, initial_state=initial_state, output_final_state=output_final_state, offsets=offsets, head_first=head_first, chunk_size=BT)
    o = chunk_dplr_fwd_o(qg=qg, v=v, v_new=v_new, A_qk=A_qk, A_qb=A_qb, h=h, offsets=offsets, indices=indices, head_first=head_first, chunk_size=BT)
    return o, final_state


def chunk_dplr_delta_rule(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', a: 'torch.Tensor', b: 'torch.Tensor', gk: 'torch.Tensor', scale: 'Optional[float]'=None, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=False):
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            activations of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            betas of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        gk (torch.Tensor):
            gk of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`. decay term in log space!
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDPLRDeltaRuleFunction.apply(q, k, v, a, b, gk, scale, initial_state, output_final_state, cu_seqlens, head_first)
    return o, final_state


def chunk_rwkv7(r: 'torch.Tensor', log_w: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', a: 'torch.Tensor', b: 'torch.Tensor', scale: 'float'=1.0, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=True, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=False):
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        log_w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """
    return chunk_dplr_delta_rule(q=r, k=k, v=v, a=a, b=b, gk=log_w, scale=scale, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens, head_first=head_first)


class FusedRecurrentDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, a, b, gk, scale=None, initial_state=None, output_final_state=False, offsets=None, head_first=False):
        if head_first:
            B, H, T, K, V = *k.shape, v.shape[-1]
        else:
            B, T, H, K, V = *k.shape, v.shape[-1]
        N = B if offsets is None else len(offsets) - 1
        BK = triton.next_power_of_2(K)
        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
        else:
            final_state = None
        ha = torch.empty_like(v, dtype=torch.float32)

        def grid(meta):
            return triton.cdiv(V, meta['BV']), N * H
        o = torch.empty_like(v)
        fused_recurrent_fwd_kernel[grid](q=q, k=k, v=v, a=a, b=b, gk=gk, o=o, ha=ha, h0=initial_state, ht=final_state, scale=scale, offsets=offsets, H=H, T=T, K=K, V=V, BK=BK, HEAD_FIRST=head_first)
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        raise NotImplementedError('Backward pass for fused_recurrent_dplr_delta_rule is not implemented and will not be supported. This kernel is only for inference. For training, please use `chunk_dplr_delta_rule`.')


def fused_recurrent_dplr_delta_rule(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', a: 'torch.Tensor', b: 'torch.Tensor', gk: 'torch.Tensor', scale: 'Optional[float]'=1.0, initial_state: 'Optional[torch.Tensor]'=None, output_final_state: 'bool'=False, cu_seqlens: 'Optional[torch.Tensor]'=None, head_first: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes the recurrence S_t = S_t @ (I + a_t b_t^T) + v_t k_t^T in a recurrent manner.

    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]`
        a (torch.Tensor):
            as of shape `[B, H, T, K]`
        b (torch.Tensor):
             bs of shape `[B, H, T, K]`
        gk (torch.Tensor):
            gk of shape `[B, H, T, K]`
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If None, it will default to `1 / sqrt(K)`. Default: `1.0`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape `[N + 1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.Please flatten variable-length inputs before processing.')
        if head_first:
            raise RuntimeError('Sequences with variable lengths are not supported for head-first mode')
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f'The number of initial states is expected to be equal to the number of input sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.')
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, 'scale must be positive'
    o, final_state = FusedRecurrentDPLRDeltaRuleFunction.apply(q, k, v, a, b, gk, scale, initial_state, output_final_state, cu_seqlens, head_first)
    return o, final_state


def fused_recurrent_rwkv7(r: 'torch.Tensor', log_w: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', a: 'torch.Tensor', b: 'torch.Tensor', scale: 'float'=1.0, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=True, cu_seqlens: 'Optional[torch.LongTensor]'=None, head_first: 'bool'=False):
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        log_w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (torch.Tensor):
            initial state of shape `[B, H, K, V]` if cu_seqlens is None else `[N, H, K, V]` where N = len(cu_seqlens) - 1.
        output_final_state (bool):
            whether to output the final state.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """
    return fused_recurrent_dplr_delta_rule(q=r, k=k, v=v, a=a, b=b, gk=log_w, scale=scale, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens, head_first=head_first)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, eps=1e-06, output_dtype=None):
        y = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy, *args):
        x, = ctx.saved_tensors
        dx = l2norm_bwd(x, dy, ctx.eps)
        return dx, None, None


def l2_norm(x: 'torch.Tensor', eps: 'float'=1e-06, output_dtype: 'Optional[torch.dtype]'=None) ->torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


swiglu_bwd_with_output_codestring = """
template <typename T> T swiglu_bwd_with_output(T x, T y, T g, T& dx, T& dy, T& z) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    float x_swish = float(x) * x_sigmoid;
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = x_swish * float(g);
    z = x_swish * float(y);
}
"""


swiglu_bwd_with_output = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_with_output_codestring, num_outputs=3)


class ABCBlock(nn.Module):

    def __init__(self, config: 'ABCConfig', layer_idx: 'int'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(hidden_size=config.hidden_size, num_heads=config.attn['num_heads'], num_kv_heads=config.attn['num_kv_heads'], window_size=config.attn['window_size'], rope_theta=config.attn['rope_theta'], max_position_embeddings=config.max_position_embeddings, layer_idx=layer_idx)
        else:
            self.attn = ABCAttention(hidden_size=config.hidden_size, expand_k=config.expand_k, expand_v=config.expand_v, num_heads=config.num_heads, num_slots=config.num_slots, use_short_conv=config.use_short_conv, conv_size=config.conv_size, gate_fn=config.hidden_act, elementwise_affine=config.elementwise_affine, norm_eps=config.norm_eps, clamp_min=config.clamp_min, clamp_max=config.clamp_max, fuse_norm=config.fuse_norm, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = ABCMLP(hidden_size=config.hidden_size, hidden_ratio=config.hidden_ratio, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, past_key_values: 'Optional[Union[Cache, List[torch.FloatTensor]]]'=None, use_cache: 'Optional[bool]'=False, output_attentions: 'Optional[bool]'=False, **kwargs) ->Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(hidden_states=hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions)
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states, attentions, past_key_values
        return outputs


class LinearAttentionBlock(nn.Module):

    def __init__(self, config: 'LinearAttentionConfig', layer_idx: 'int'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(hidden_size=config.hidden_size, num_heads=config.attn['num_heads'], num_kv_heads=config.attn['num_kv_heads'], window_size=config.attn['window_size'], rope_theta=config.attn['rope_theta'], max_position_embeddings=config.max_position_embeddings, layer_idx=layer_idx)
        else:
            self.attn = LinearAttention(mode=config.attn_mode, hidden_size=config.hidden_size, expand_k=config.expand_k, expand_v=config.expand_v, num_heads=config.num_heads, num_kv_heads=config.num_kv_heads, feature_map=config.feature_map, tie_feature_map_qk=config.tie_feature_map_qk, norm_q=config.norm_q, norm_k=config.norm_k, do_feature_map_norm=config.norm_feature_map, elementwise_affine=config.elementwise_affine, norm_eps=config.norm_eps, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = LinearAttentionMLP(hidden_size=config.hidden_size, hidden_ratio=config.hidden_ratio, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, past_key_values: 'Optional[Union[Cache, List[torch.FloatTensor]]]'=None, use_cache: 'Optional[bool]'=False, output_attentions: 'Optional[bool]'=False, **kwargs) ->Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        attn_weights, present_key_value = None, None
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        if use_cache:
            outputs += present_key_value,
        return outputs


logger = logging.get_logger(__name__)


class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: 'MambaConfig', layer_idx: 'int'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(in_channels=self.intermediate_size, out_channels=self.intermediate_size, bias=config.use_conv_bias, kernel_size=config.conv_kernel, groups=self.intermediate_size, padding=config.conv_kernel - 1)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias
        if not is_fast_path_available:
            logger.warning_once('The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)` is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and https://github.com/Dao-AILab/causal-conv1d')

    def cuda_kernels_forward(self, hidden_states: 'torch.Tensor', cache_params: 'Optional[MambaCache]'=None, cache_position: 'Optional[torch.LongTensor]'=None, attention_mask: 'Optional[torch.LongTensor]'=None):
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        if self.training and cache_params is None:
            contextualized_states = mamba_inner_fn(projected_states, self.conv1d.weight, self.conv1d.bias if self.use_conv_bias else None, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias.float() if self.use_bias else None, -torch.exp(self.A_log.float()), None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_position[0] > 0:
                hidden_states = causal_conv1d_update(hidden_states.squeeze(-1), cache_params.conv_states[self.layer_idx], conv_weights, self.conv1d.bias, self.activation)
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                    cache_params.update_conv_state(self.layer_idx, conv_states, cache_position)
                hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
            A = -torch.exp(self.A_log.float())
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, 'bias') else None
            if cache_params is not None and cache_position[0] > 0:
                scan_outputs = selective_state_update(cache_params.ssm_states[self.layer_idx], hidden_states[..., 0], discrete_time_step[..., 0], A, B[:, 0], C[:, 0], self.D, gate[..., 0], time_proj_bias, dt_softplus=True).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(hidden_states, discrete_time_step, A, B.transpose(1, 2), C.transpose(1, 2), self.D.float(), gate, time_proj_bias, delta_softplus=True, return_last_state=True)
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(self.layer_idx, ssm_state)
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def slow_forward(self, input_states, cache_params: 'Optional[MambaCache]'=None, cache_position: 'Optional[torch.LongTensor]'=None, attention_mask: 'Optional[torch.LongTensor]'=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state
            if cache_position.shape[0] == self.conv_kernel_size:
                conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
            else:
                conv_state = cache_params.update_conv_state(self.layer_idx, hidden_states, cache_position)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).unsqueeze(-1)
        else:
            ssm_state = torch.zeros((batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state, C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + hidden_states * self.D[None, :, None]
        scan_output = scan_output * self.act(gate)
        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))
        return contextualized_states

    def forward(self, hidden_states, cache_params: 'Optional[MambaCache]'=None, cache_position: 'Optional[torch.LongTensor]'=None, attention_mask: 'Optional[torch.LongTensor]'=None):
        if is_fast_path_available and 'cuda' in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        return self.slow_forward(hidden_states, cache_params, cache_position, attention_mask)


class MambaBlock(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, cache_params: 'Optional[MambaCache]'=None, cache_position: 'Optional[torch.LongTensor]'=None, attention_mask: 'Optional[torch.LongTensor]'=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual
        hidden_states = self.mixer(hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        if self.residual_in_fp32:
            hidden_states = hidden_states
        return hidden_states


class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, z=None, eps=1e-06, group_size=None, norm_before_gate=True, is_rms_norm=False):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = layer_norm_fwd(x, weight, bias, eps, z=z, group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm)
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd, z = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        dx, dw, db, dz = layer_norm_bwd(dy, x, weight, bias, ctx.eps, mean, rstd, z, ctx.group_size, ctx.norm_before_gate, ctx.is_rms_norm)
        dx = dx.reshape(ctx.x_shape_og)
        dx = dz.reshape(ctx.x_shape_og) if dz is not None else None
        return dx, dw, db, dz, None, None, None, None


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-06, group_size=None, norm_before_gate=True):
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, True)


class RMSNormGated(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-05, group_size=None, norm_before_gate=False, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter('bias', None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return rmsnorm_fn(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size, norm_before_gate=self.norm_before_gate)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = hidden_states * attention_mask[:, :, None]
    return hidden_states


def pad_tensor_by_size(input_tensor: 'torch.Tensor', pad_size: 'int'):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return torch.nn.functional.pad(input_tensor, pad_shape, mode='constant', value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3])


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: 'Mamba2Config', layer_idx: 'int'):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm
        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size
        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=config.use_conv_bias, kernel_size=config.conv_kernel, groups=self.conv_dim, padding=config.conv_kernel - 1)
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = RMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon, norm_before_gate=False)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias
        if not is_fast_path_available:
            logger.warning_once('The fast path is not available because one of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)` is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation andhttps://github.com/Dao-AILab/causal-conv1d')

    def cuda_kernels_forward(self, hidden_states: 'torch.Tensor', cache_params: 'Optional[Mamba2Cache]'=None, cache_position: 'Optional[torch.LongTensor]'=None, attention_mask: 'Optional[torch.Tensor]'=None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            _, _, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split([d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
            hidden_states_B_C = causal_conv1d_update(hidden_states_B_C, cache_params.conv_states[self.layer_idx], self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
            hidden_states, B, C = torch.split(hidden_states_B_C, [self.intermediate_size, groups_time_state_size, groups_time_state_size], dim=-1)
            A = -torch.exp(self.A_log.float())
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(cache_params.ssm_states[self.layer_idx], hidden_states_reshaped, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True)
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        else:
            A = -torch.exp(self.A_log.float())
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float('inf')) else {'dt_limit': self.time_step_limit}
            if self.training and cache_params is None:
                out = mamba_split_conv1d_scan_combined(projected_states, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.dt_bias, A, D=self.D, chunk_size=self.chunk_size, seq_idx=None, activation=self.activation, rmsnorm_weight=self.norm.weight, rmsnorm_eps=self.norm.eps, outproj_weight=self.out_proj.weight, outproj_bias=self.out_proj.bias, headdim=self.head_dim, ngroups=self.n_groups, norm_before_gate=False, return_final_states=False, **dt_limit_kwargs)
            else:
                _, _, gate, hidden_states_B_C, dt = projected_states.split([d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
                if cache_params is not None:
                    hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                    conv_states = nn.functional.pad(hidden_states_B_C_transposed, (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0))
                    cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True)
                if self.activation not in ['silu', 'swish']:
                    hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
                else:
                    hidden_states_B_C = causal_conv1d_fn(x=hidden_states_B_C.transpose(1, 2), weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias, activation=self.activation).transpose(1, 2)
                hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
                hidden_states, B, C = torch.split(hidden_states_B_C, [self.intermediate_size, groups_time_state_size, groups_time_state_size], dim=-1)
                scan_output, ssm_state = mamba_chunk_scan_combined(hidden_states.view(batch_size, seq_len, -1, self.head_dim), dt, A, B.view(batch_size, seq_len, self.n_groups, -1), C.view(batch_size, seq_len, self.n_groups, -1), chunk_size=self.chunk_size, D=self.D, z=None, seq_idx=None, return_final_states=True, dt_bias=self.dt_bias, dt_softplus=True, **dt_limit_kwargs)
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)
        return out

    def torch_forward(self, input_states, cache_params: 'Optional[Mamba2Cache]'=None, cache_position: 'Optional[torch.LongTensor]'=None, attention_mask: 'Optional[torch.Tensor]'=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        projected_states = self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split([d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False)
            conv_states = cache_params.conv_states[self.layer_idx]
            hidden_states_B_C = torch.sum(conv_states * self.conv1d.weight.squeeze(1), dim=-1)
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = nn.functional.pad(hidden_states_B_C_transposed, (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0))
                cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True)
            hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(hidden_states_B_C, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)
        A = -torch.exp(self.A_log.float())
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            cache_device = cache_params.ssm_states.device
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
            dt = torch.nn.functional.softplus(dt + dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size)
            dA = torch.exp(dt[..., None] * A)
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            dB = dt[..., None] * B[..., None, :]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = dB * hidden_states[..., None]
            cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=cache_params.ssm_states[self.layer_idx] * dA + dBx)
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            ssm_states = cache_params.ssm_states[self.layer_idx]
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = y + hidden_states * D
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)
            hidden_states = hidden_states * dt[..., None]
            A = A * dt
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)
            L = torch.exp(segment_sum(A))
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
            G = G_intermediate.sum(dim=-1)
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)
            decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)
            if cache_params is not None and cache_position is not None and cache_position[0] > 0:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...]
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]
            y = Y_diag + Y_off
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
            y = y + D_residual
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)
        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output)
        return contextualized_states

    def forward(self, hidden_states, cache_params: 'Optional[Mamba2Cache]'=None, cache_position: 'Optional[torch.LongTensor]'=None, attention_mask: 'Optional[torch.Tensor]'=None):
        if is_fast_path_available and 'cuda' in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[:, :, None]
        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)


class Mamba2Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, cache_params: 'Optional[Mamba2Cache]'=None, cache_position: 'Optional[torch.LongTensor]'=None, attention_mask: 'Optional[torch.Tensor]'=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual
        hidden_states = self.mixer(hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        if self.residual_in_fp32:
            hidden_states = hidden_states
        return hidden_states


class RWKV6Block(nn.Module):

    def __init__(self, config: 'RWKV6Config', layer_idx: 'int'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        if config.norm_first and layer_idx == 0:
            self.pre_norm = LayerNorm(hidden_size=config.hidden_size, bias=config.norm_bias, eps=config.norm_eps)
        self.attn_norm = LayerNorm(hidden_size=config.hidden_size, bias=config.norm_bias, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(hidden_size=config.hidden_size, num_heads=config.attn['num_heads'], num_kv_heads=config.attn['num_kv_heads'], window_size=config.attn['window_size'], rope_theta=config.attn['rope_theta'], max_position_embeddings=config.max_position_embeddings, layer_idx=layer_idx)
        else:
            self.attn = RWKV6Attention(mode=config.attn_mode, hidden_size=config.hidden_size, expand_k=config.expand_k, expand_v=config.expand_v, num_heads=config.num_heads, proj_low_rank_dim=config.proj_low_rank_dim, gate_low_rank_dim=config.gate_low_rank_dim, norm_eps=config.norm_eps, fuse_norm=config.fuse_norm, layer_idx=layer_idx)
        self.ffn_norm = LayerNorm(hidden_size=config.hidden_size, bias=config.norm_bias, eps=config.norm_eps)
        self.ffn = RWKV6FeedForward(hidden_size=config.hidden_size, hidden_ratio=config.hidden_ratio, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, layer_idx=layer_idx)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, past_key_values: 'Optional[Cache]'=None, use_cache: 'Optional[bool]'=False, output_attentions: 'Optional[bool]'=False, **kwargs) ->Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states
        hidden_states = self.attn_norm(residual)
        hidden_states, attentions, past_key_values = self.attn(hidden_states=hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, **kwargs)
        hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
        hidden_states, past_key_values = self.ffn(hidden_states, attention_mask, past_key_values)
        hidden_states = residual + hidden_states
        outputs = hidden_states, attentions, past_key_values
        return outputs


def fft_conv(u, k, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u, n=fft_size)
    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
    out = y + u
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return out * rearrange(dropout_mask, 'b H -> b H 1')
    else:
        return out


class LongConvolution(nn.Module):
    """
    LongConvolution applies a convolution operation on the input tensor using a fixed
    filter of length max_len.
    The filter is learned during training and is applied using FFT convolution.
    Args:
        hidden_size (int): The number of expected features in the input and output.
        max_len (int): The maximum sequence length.
    Returns:
        y: [batch_size, seq_len, hidden_size] tensor
    """

    def __init__(self, hidden_size: 'int', max_len: 'int', **kwargs):
        """
        Initializes the LongConvolution module.
        Args:
            hidden_size (int): The number of expected features in the input and output.
            max_len (int): The maximum sequence length.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.filter = nn.Parameter(torch.randn(self.hidden_size, max_len), requires_grad=True)

    def forward(self, x: 'torch.Tensor', *args, **kwargs):
        """
        Applies the LongConvolution operation on the input tensor.
        Args:
            x: [batch_size, seq_len, hidden_size] tensor
        Returns:
            y: [batch_size, seq_len, hidden_size] tensor
        """
        x = x.transpose(1, 2)
        y = fft_conv(x, self.filter, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y


class PositionalEmbedding(nn.Module):

    def __init__(self, emb_dim: 'int', seq_len: 'int', **kwargs):
        """Complex exponential positional embeddings for implicit long convolution filters."""
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        f = torch.linspace(0.0001, bands - 1, bands)[None, None]
        z = torch.exp(-1.0j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L]


class ImplicitLongConvolution(nn.Module):
    """
    Long convolution with implicit filter parameterized by an MLP.

    Args:
        hidden_size (int):
            The number of expected features in the input and output.
        max_len (int):
            The maximum sequence length.
        d_emb (Optional[int]):
            The dimension of the positional embeddings. Must be odd and greater or equal to 3 (time, sine and cosine).
            Defaults to 3.
        d_hidden (Optional[int]):
            The number of features in the hidden layer of the MLP. Defaults to 16.

    Attributes:
        pos_emb (`PositionalEmbedding`): The positional embedding layer.
        mlp (`nn.Sequential`): The MLP that parameterizes the implicit filter.

    """

    def __init__(self, hidden_size: 'int', max_len: 'int', d_emb: 'int'=3, d_hidden: 'int'=16, **kwargs):
        """
        Long convolution with implicit filter parameterized by an MLP.


        """
        super().__init__()
        self.hidden_size = hidden_size
        self.d_emb = d_emb
        assert d_emb % 2 != 0 and d_emb >= 3, 'd_emb must be odd and greater or equal to 3 (time, sine and cosine)'
        self.pos_emb = PositionalEmbedding(d_emb, max_len)
        self.mlp = nn.Sequential(nn.Linear(d_emb, d_hidden), torch.nn.ReLU(), nn.Linear(d_hidden, hidden_size))

    def filter(self, seq_len: 'int', *args, **kwargs):
        k = self.mlp(self.pos_emb(seq_len))
        return k.transpose(1, 2)

    def forward(self, x: 'torch.Tensor', *args, **kwargs):
        """
        Args:
            x: [batch_size, seq_len, hidden_size] tensor
        Returns:
            y: [batch_size, seq_len, hidden_size] tensor
        """
        x = x.transpose(1, 2)
        k = self.filter(x.shape[-1])
        y = fft_conv(x, k, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y


@checkpoint
def flatten_diag_outer_product(x, y):
    z = torch.einsum('...i,...j->...ij', x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N)
    return z[..., indicies[0], indicies[1]]


def is_power_of_2(n):
    return n & n - 1 == 0 and n != 0


def fused_cross_entropy_forward(logits: 'torch.Tensor', target: 'torch.Tensor', label_smoothing: 'float'=0.0, logit_scale: 'float'=1.0, lse_square_scale: 'float'=0.0, ignore_index: 'int'=-100, process_group=None):
    n_rows, n_cols = logits.shape
    assert target.shape == (n_rows,)
    world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
    total_classes = world_size * n_cols
    rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
    class_start_idx = rank * n_cols
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    MAX_BLOCK_SIZE = 64 * 1024
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 128 * 1024 else 32
    split = world_size > 1 or n_cols > MAX_BLOCK_SIZE
    n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
    losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    with torch.device(logits.device.index):
        cross_entropy_fwd_kernel[n_rows, n_splits](losses, lse, z_losses, logits, target, label_smoothing, logit_scale, lse_square_scale, ignore_index, total_classes, class_start_idx, n_cols, n_rows, logits.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, SPLIT=split)
    if split:
        if n_splits > 1:
            lse = torch.logsumexp(lse, dim=0)
            losses = losses.sum(dim=0)
        if world_size > 1:
            lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
            handle_losses = torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True)
            lse = torch.logsumexp(lse_allgather, dim=0)
            handle_losses.wait()
        losses += lse
        if lse_square_scale != 0.0:
            z_losses = lse_square_scale * lse.square()
            z_losses.masked_fill_(target == ignore_index, 0.0)
            losses += z_losses
        else:
            z_losses = torch.zeros_like(losses)
        losses.masked_fill_(target == ignore_index, 0.0)
    return losses, z_losses, lse, total_classes, class_start_idx


class CrossEntropyLossFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, logits, target, label_smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignore_index=-100, inplace_backward=False, process_group=None):
        losses, z_losses, lse, total_classes, class_start_idx = fused_cross_entropy_forward(logits, target, label_smoothing, logit_scale, lse_square_scale, ignore_index, process_group)
        ctx.save_for_backward(logits, lse, target)
        ctx.mark_non_differentiable(z_losses)
        ctx.label_smoothing = label_smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignore_index = ignore_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward
        return losses, z_losses

    @staticmethod
    @contiguous
    def backward(ctx, grad_losses, grad_z_losses):
        del grad_z_losses
        logits, lse, target = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16

        def grid(META):
            return n_rows, triton.cdiv(n_cols, META['BLOCK_SIZE'])
        with torch.device(logits.device.index):
            cross_entropy_bwd_kernel[grid](dlogits, grad_losses, logits, lse, target, ctx.label_smoothing, ctx.logit_scale, ctx.lse_square_scale, ctx.ignore_index, ctx.total_classes, ctx.class_start_idx, n_cols, logits.stride(0), dlogits.stride(0), grad_losses.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        return dlogits, None, None, None, None, None, None, None, None


def cross_entropy_loss(logits: 'torch.Tensor', target: 'torch.Tensor', label_smoothing: 'float'=0.0, logit_scale: 'float'=1.0, lse_square_scale: 'float'=0.0, ignore_index=-100, inplace_backward: 'bool'=False, process_group=None) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        logits: [batch, vocab_size]
        target: [batch,]
        label_smoothing: float
        logit_scale: float.
            Multiply logits by this scale before calculating the loss.
        lse_square_scale: float.
            If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
            This is also referred to as "z-loss".
        ignore_index: int.
            If target == ignore_index, the loss is set to 0.0.
        inplace_backward: bool.
            If True, we do the backward pass in-place by modifying the logits.
            This saves memory.
        process_group:
            if not None, we're doing Tensor Parallel: each process is responsible for
            one part of the vocab. The loss will be aggregated across processes.
    Returns:
        losses: [batch,], float
        z_losses: [batch,], float
    """
    return CrossEntropyLossFunction.apply(logits, target, label_smoothing, logit_scale, lse_square_scale, ignore_index, inplace_backward, process_group)


class FusedCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index: 'int'=-100, reduction: 'str'='mean', label_smoothing: 'float'=0.0, logit_scale: 'float'=1.0, lse_square_scale: 'float'=0.0, inplace_backward: 'bool'=False, process_group: 'Any'=None, return_z_loss: 'bool'=False):
        """
        Arguments:
            ignore_index: int. If target == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            lse_square_scale: float. If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
                This is also referred to as "z-loss".
            inplace_backward: bool. If True, we do the backward pass in-place by modifying the logits.
                This saves memory.
            process_group: if not None, we're doing Tensor Parallel: each process is responsible for
                one part of the vocab. The loss will be aggregated across processes.
            return_z_loss: bool. If True, we return the component of the loss contributed by
                the lse_square_scale value. This value is only for logging and does not support
                backprop.
        """
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError("Only support reduction = 'mean' or 'none' or 'sum'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.inplace_backward = inplace_backward
        self.process_group = process_group
        self.return_z_loss = return_z_loss

    def forward(self, input, target):
        """
        Arguments:
            input: (batch, vocab_size)
            target: (batch,)
        Returns:
            losses: (batch,) if reduction is 'none', else (1,), dtype float
            z_loss: (batch,) if reduction is 'none', else (1,), dtype float (if self.return_z_loss)
        """
        loss, z_loss = cross_entropy_loss(input, target, label_smoothing=self.label_smoothing, logit_scale=self.logit_scale, lse_square_scale=self.lse_square_scale, ignore_index=self.ignore_index, inplace_backward=self.inplace_backward, process_group=self.process_group)
        if self.reduction == 'mean':
            loss = loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss
        if not self.return_z_loss:
            return loss
        if self.reduction == 'mean':
            z_loss = z_loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == 'sum':
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss
        return loss, z_loss


MAX_FUSED_SIZE = 65536 // 2


def fused_kl_div_backward(do: 'torch.Tensor', dx: 'torch.Tensor', dw: 'torch.Tensor'):
    if torch.ne(do, torch.tensor(1.0, device=do.device)):
        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
        elementwise_mul_kernel[triton.cdiv(N * H, B),](x=dx, g=do, N=N * H, B=B, num_warps=32)
        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[triton.cdiv(V * H, B),](x=dw, g=do, N=V * H, B=B, num_warps=32)
    return dx, dw


def fused_kl_div_forward(x: 'torch.Tensor', target_x: 'torch.Tensor', weight: 'torch.Tensor', target_weight: 'torch.Tensor', reduction: 'str'='batchmean'):
    device = x.device
    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    NC = min(8, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)
    dx = torch.zeros_like(x, device=device)
    dw = torch.zeros_like(weight, device=device) if weight is not None else None
    loss = torch.zeros(N, dtype=torch.float32, device=device)
    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_sx = x[start:end]
        c_tx = target_x[start:end]
        c_sl = F.linear(c_sx, weight)
        c_tl = F.linear(c_tx, target_weight)
        c_loss = loss[start:end]
        kl_div_kernel[c_sx.shape[0],](logits=c_sl, target_logits=c_tl, loss=c_loss, s_logits=c_sl.stride(-2), s_loss=c_loss.stride(-1), reduction=reduction, N=N, V=V, BV=BV, num_warps=32)
        dx[start:end] = torch.mm(c_sl, weight)
        if weight is not None:
            torch.addmm(input=dw, mat1=c_sl.t(), mat2=c_sx, out=dw)
    loss = loss.sum()
    return loss, dx, dw


class FusedKLDivLossFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x: 'torch.Tensor', target_x: 'torch.Tensor', weight: 'torch.Tensor', target_weight: 'torch.Tensor', reduction: 'str'):
        loss, dx, dw = fused_kl_div_forward(x=x, target_x=target_x, weight=weight, target_weight=target_weight, reduction=reduction)
        ctx.save_for_backward(dx, dw)
        return loss

    @staticmethod
    @contiguous
    def backward(ctx, do):
        dx, dw = ctx.saved_tensors
        dx, dw = fused_kl_div_backward(do, dx, dw)
        return dx, None, dw, None, None


def fused_kl_div_loss(x: 'torch.Tensor', target_x: 'torch.Tensor', weight: 'torch.Tensor', target_weight: 'torch.Tensor', reduction: 'str'='batchmean') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target_x (torch.Tensor): [batch_size * seq_len, hidden_size]
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        target_weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        reduction:
            Specifies the reduction to apply to the output: 'batchmean'. Default: 'batchmean'.
    Returns:
        loss
    """
    return FusedKLDivLossFunction.apply(x, target_x, weight, target_weight, reduction)


class FusedKLDivLoss(nn.Module):

    def __init__(self, reduction: 'str'='batchmean'):
        """
        Args:
            reduction:
                Specifies the reduction to apply to the output: 'batchmean'. Default: 'batchmean'.
        """
        super().__init__()
        assert reduction in ['batchmean'], f'reduction: {reduction} is not supported'
        self.reduction = reduction

    def forward(self, x: 'torch.Tensor', target_x: 'torch.Tensor', weight: 'torch.Tensor', target_weight: 'torch.Tensor'):
        """
        Args:
            x (torch.Tensor): [batch_size * seq_len, hidden_size]
            target_x (torch.Tensor): [batch_size * seq_len, hidden_size]
            weight (torch.Tensor): [vocab_size, hidden_size]
                where `vocab_size` is the number of classes.
            target_weight (torch.Tensor): [vocab_size, hidden_size]
                where `vocab_size` is the number of classes.
        Returns:
            loss
        """
        loss = fused_kl_div_loss(x=x, target_x=target_x, weight=weight, target_weight=target_weight, reduction=self.reduction)
        return loss


def fused_linear_cross_entropy_backward(do: 'torch.Tensor', dx: 'torch.Tensor', dw: 'torch.Tensor', db: 'torch.Tensor'):
    if torch.ne(do, torch.tensor(1.0, device=do.device)):
        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
        elementwise_mul_kernel[triton.cdiv(N * H, B),](x=dx, g=do, N=N * H, B=B, num_warps=32)
        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[triton.cdiv(V * H, B),](x=dw, g=do, N=V * H, B=B, num_warps=32)
        if db is not None:
            V = db.shape[0]
            elementwise_mul_kernel[triton.cdiv(V, B),](x=db, g=do, N=V, B=B, num_warps=32)
    return dx, dw, db


def logsumexp_fwd(x, scale: 'Optional[float]'=None, dtype: 'Optional[torch.dtype]'=None):
    """
    Compute the logsumexp of the input tensor over the last dimension.

    Args:
        x (Tensor):
            The input tensor of any shape.
        scale (Optional[float]):
            The scale applied to the input tensor. Default: `None`.
        dtype (Optional[torch.dtype]):
            The data type of the output tensor. Default: `None`.
    Returns:
        Tensor: The logsumexp of the input tensor.
    """
    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)
    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[N, ND](x=x, z=z, scale=scale, D=D, B=B)
    z = z.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z
    return z


def fused_linear_cross_entropy_forward(x: 'torch.Tensor', target: 'torch.LongTensor', weight: 'torch.Tensor', bias: 'torch.Tensor'=None, ignore_index: 'int'=-100, label_smoothing: 'float'=0.0, logit_scale: 'float'=1.0, num_chunks: 'int'=8, reduction: 'str'='mean'):
    device = x.device
    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)
    dx = torch.zeros_like(x, device=device)
    dw = torch.zeros_like(weight, device=device, dtype=torch.float) if weight is not None else None
    db = torch.zeros_like(bias, device=device, dtype=torch.float) if bias is not None else None
    loss = torch.zeros(N, device=device, dtype=torch.float)
    total = target.ne(ignore_index).sum().item()
    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_x = x[start:end]
        c_logits = F.linear(c_x, weight, bias)
        c_target = target[start:end]
        c_lse = logsumexp_fwd(c_logits, scale=logit_scale, dtype=torch.float)
        c_loss = loss[start:end]
        cross_entropy_kernel[c_logits.shape[0],](logits=c_logits, lse=c_lse, target=c_target, loss=c_loss, total=total, ignore_index=ignore_index, label_smoothing=label_smoothing, logit_scale=logit_scale, reduction=reduction, V=V, BV=BV, num_warps=32)
        dx[start:end] = torch.mm(c_logits, weight)
        if weight is not None:
            dw += c_logits.t() @ c_x
        if bias is not None:
            torch.add(input=db, other=c_logits.sum(0), out=db)
    loss = loss.sum()
    if dw is not None:
        dw = dw
    if db is not None:
        db = db
    return loss, dx, dw, db


class FusedLinearCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x: 'torch.Tensor', target: 'torch.LongTensor', weight: 'torch.Tensor', bias: 'torch.Tensor'=None, ignore_index: 'int'=-100, label_smoothing: 'float'=0.0, logit_scale: 'float'=1.0, num_chunks: 'int'=8, reduction: 'str'='mean'):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the x and target
        for the backward pass.

        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index:
            the index to ignore in the target.
        label_smoothing:
            the amount of smoothing when computing the loss, where 0.0 means no smoothing.
        logit_scale: float = 1.0,
            A scaling factor applied to the logits. Default: 1.0
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
        """
        loss, dx, dw, db = fused_linear_cross_entropy_forward(x, target, weight, bias, ignore_index, label_smoothing, logit_scale, num_chunks, reduction)
        ctx.save_for_backward(dx.detach(), dw.detach() if weight is not None else None, db.detach() if bias is not None else None)
        return loss

    @staticmethod
    @contiguous
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        dx, dw, db = fused_linear_cross_entropy_backward(do, dx, dw, db)
        return dx, None, dw, db, None, None, None, None, None


def fused_linear_cross_entropy_loss(x: 'torch.Tensor', target: 'torch.LongTensor', weight: 'torch.Tensor', bias: 'torch.Tensor'=None, ignore_index: 'int'=-100, label_smoothing: 'float'=0.0, logit_scale: 'float'=1.0, num_chunks: 'int'=8, reduction: 'str'='mean') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index: int.
            If target == ignore_index, the loss is set to 0.0.
        label_smoothing: float
        logit_scale: float
            A scaling factor applied to the logits. Default: 1.0
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
    Returns:
        losses: [batch,], float
    """
    return FusedLinearCrossEntropyFunction.apply(x, target, weight, bias, ignore_index, label_smoothing, logit_scale, num_chunks, reduction)


class FusedLinearCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index: 'int'=-100, label_smoothing: 'float'=0.0, logit_scale: 'float'=1.0, num_chunks: 'int'=8, reduction: 'str'='mean'):
        """
        Args:
            ignore_index: int.
                If target == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            logit_scale: float
                A scaling factor applied to the logits. Default: 1.0
            num_chunks: int
                The number of chunks to split the input tensor into for processing.
                This can help optimize memory usage and computation speed.
                Default: 8
            reduction:
                Specifies the reduction to apply to the output: 'mean' | 'sum'.
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed.
                Default: 'mean'.
        """
        super().__init__()
        assert reduction in ['none', 'mean', 'sum'], f'reduction: {reduction} is not supported'
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.num_chunks = num_chunks
        self.reduction = reduction

    def forward(self, x: 'torch.Tensor', target: 'torch.LongTensor', weight: 'torch.Tensor', bias: 'Optional[torch.Tensor]'=None):
        """
        Args:
            x (torch.Tensor): [batch_size * seq_len, hidden_size]
            target (torch.LongTensor): [batch_size * seq_len]
                where each value is in [0, V).
            weight (torch.Tensor): [vocab_size, hidden_size]
                where `vocab_size` is the number of classes.
            bias (Optional[torch.Tensor]): [vocab_size]
                where `vocab_size` is the number of classes.
        Returns:
            loss
        """
        loss = fused_linear_cross_entropy_loss(x, target, weight=weight, bias=bias, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing, logit_scale=self.logit_scale, num_chunks=self.num_chunks, reduction=self.reduction)
        return loss


def layer_norm_swish_gate_fn(x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-06):
    return LayerNormSwishGateFn.apply(x, o, weight, bias, residual, eps, prenorm, residual_in_fp32, False)


def layer_norm_swish_gate_linear_fn(x, o, norm_weight, norm_bias, linear_weight, linear_bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-06):
    return LayerNormSwishGateLinearFn.apply(x, o, norm_weight, norm_bias, linear_weight, linear_bias, residual, eps, prenorm, residual_in_fp32, False)


def layernorm_fn(x, weight, bias, z=None, eps=1e-06, group_size=None, norm_before_gate=True, is_rms_norm=False):
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)


class LayerNormGated(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-05, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return layernorm_fn(x, self.weight, self.bias, z=z, group_size=self.group_size, eps=self.eps, norm_before_gate=self.norm_before_gate)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DDLerpLinear,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImplicitLongConvolution,
     lambda: ([], {'hidden_size': 4, 'max_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LerpLinear,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LoRA,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'low_rank_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LongConvolution,
     lambda: ([], {'hidden_size': 4, 'max_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionalEmbedding,
     lambda: ([], {'emb_dim': 4, 'seq_len': 4}),
     lambda: ([0], {}),
     True),
]

class Test_fla_org_flash_linear_attention(_paritybench_base):
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

