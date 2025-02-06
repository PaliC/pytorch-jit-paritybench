import sys
_module = sys.modules[__name__]
del sys
cache = _module
lit_gpt = _module
config = _module
fused_cross_entropy = _module
fused_rotary_embedding = _module
gated_delta_net = _module
gated_delta_rule_ops = _module
chunk = _module
wy_fast = _module
model = _module
packed_dataset = _module
rmsnorm = _module
rotary = _module
speed_monitor = _module
tokenizer = _module
utils = _module
pretrain = _module

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


from typing import Any


from typing import Literal


from typing import Optional


from typing import Type


import torch.nn as nn


import math


from typing import Tuple


from typing import TYPE_CHECKING


import torch.nn.functional as F


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from typing import List


from torch import Tensor


from functools import partial


import random


import numpy as np


from torch.utils.data import IterableDataset


from torch.utils.data import get_worker_info


from torch.nn import init


from typing import Union


import time


from collections import deque


from typing import Callable


from typing import Deque


from typing import Dict


from torch.utils.flop_counter import FlopCounterMode


import warnings


from types import MethodType


from typing import Mapping


from typing import TypeVar


import torch.utils._device


from torch.serialization import normalize_storage_type


from torch.utils.data import DataLoader


import torch.multiprocessing as mp


class SoftmaxCrossEntropyLossFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, ignored_index=-100, inplace_backward=False, process_group=None):
        """
        logits: (batch, vocab_size)
        labels: (batch,)
        If process_group is not None, we're doing Tensor Parallel: each process is responsible for
        one part of the vocab. The loss needs to be aggregated across processes.
        """
        batch, vocab_size = logits.shape
        assert labels.shape == (batch,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        ctx.total_classes = world_size * vocab_size
        if world_size == 1:
            losses, lse = xentropy_cuda_lib.forward(logits, labels, smoothing)
            losses.masked_fill_(labels == ignored_index, 0)
            labels_local = labels
        else:
            rank = torch.distributed.get_rank(process_group)
            vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size
            labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
            ignored_mask = labels == ignored_index
            labels_local = torch.where(ignored_mask, labels, labels - vocab_start_index)
            losses, lse_local = xentropy_cuda_lib.forward(logits, labels_local, smoothing, world_size * vocab_size)
            assert lse_local.shape == (batch,)
            assert losses.shape == (batch,)
            losses.masked_fill_(ignored_mask, 0)
            lse_allgather = torch.empty(world_size, batch, dtype=lse_local.dtype, device=lse_local.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse_local.contiguous(), group=process_group)
            handle_losses = torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True)
            lse = torch.logsumexp(lse_allgather, dim=0)
            rank_per_sample = torch.div(labels, vocab_size, rounding_mode='floor')
            lse_local = lse_allgather[rank_per_sample, torch.arange(batch, device=lse_allgather.device)]
            handle_losses.wait()
            if smoothing == 0.0:
                losses += lse - lse_local
            else:
                losses += (1 - smoothing) * (lse - lse_local) + smoothing * (lse - lse_allgather.sum(dim=0))
            losses.masked_fill_(ignored_mask, 0)
        ctx.save_for_backward(logits, lse, labels_local)
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits, lse, labels = ctx.saved_tensors
        grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels == ctx.ignored_index, 0)
        grad_logits = xentropy_cuda_lib.backward(grad_loss, logits, lse, labels, ctx.smoothing, ctx.inplace_backward, ctx.total_classes)
        return grad_logits, None, None, None, None, None, None


class FusedCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-100, reduction='mean', label_smoothing=0.0, inplace_backward=True, process_group=None):
        super().__init__()
        if reduction not in ['mean', 'none']:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, input, target):
        if len(input.shape) == 3:
            input = input.view(-1, input.size(-1))
            target = target.view(-1)
        loss = SoftmaxCrossEntropyLossFn.apply(input, target, self.label_smoothing, self.ignore_index, self.inplace_backward, self.process_group)
        if self.reduction == 'mean':
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: 'int', dim: 'int'=-1, eps: 'float'=1e-05) ->None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


def bwd_prepare_wy_repr(k, v, beta, g, A_w, A_u, A_w_original, A_u_original, dw, du, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT)
    dk = torch.empty_like(k).float()
    dv = torch.empty_like(v).contiguous()
    dbeta = torch.empty_like(beta).float()
    dA_w = torch.zeros_like(A_w).float()
    dA_w_original = torch.zeros_like(dA_w)
    dA_u = torch.zeros_like(A_u).float()
    dA_u_original = torch.zeros_like(dA_u)
    bwd_prepare_wy_repr_kernel_dA[NT, B * H](k, v, beta, dw, du, A_w, A_u, dA_w, dA_u, dk, dv, dbeta, k.stride(1), k.stride(2), k.stride(3), v.stride(1), v.stride(2), v.stride(3), T, K, V, BT, BK, BV)
    bwd_prepare_wy_repr_kernel_dA_recurrence[NT, B * H](A_w, A_w_original, dA_w, dA_w_original, k.stride(1), k.stride(2), k.stride(3), v.stride(1), v.stride(2), v.stride(3), T, K, V, BT, BK, BV)
    bwd_prepare_wy_repr_kernel_dA_recurrence[NT, B * H](A_u, A_u_original, dA_u, dA_u_original, k.stride(1), k.stride(2), k.stride(3), v.stride(1), v.stride(2), v.stride(3), T, K, V, BT, BK, BV)
    dk2 = torch.empty_like(k).float()
    dg = torch.empty_like(g).float()
    dbeta2 = torch.empty_like(beta).float()
    bwd_prepare_wy_repr_dk_dbeta_dg[NT, B * H](k, beta, g, dA_w_original, dA_u_original, A_w_original, dk2, dbeta2, dg, k.stride(1), k.stride(2), k.stride(3), v.stride(1), v.stride(2), v.stride(3), T, K, V, BT, BK, BV)
    dk = dk + dk2
    dbeta = dbeta + dbeta2
    return dk, dv, dbeta, dg


def chunk_bwd_dhu_fn(q, k, w, g, do, dv, BT):
    B, H, T, K, V = *q.shape, do.shape[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, 'current kernel does not support head dimension being larger than 256.'
    BV = 16 if BK > 128 else 32
    BV = 64 if BK <= 64 else BV
    BC = 16 if BK > 128 else 32
    BC = 64 if BK <= 64 else BC
    BC = min(BT, BC)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    dh = q.new_empty(B, H, NT * K, V, dtype=torch.float32)
    grid = NK, NV, B * H
    dv2 = torch.empty_like(dv)
    chunk_gated_delta_rule_bwd_kernel_dhu[grid](q, k, w, g, do, dh, dv, dv2, q.stride(1), q.stride(2), q.stride(3), do.stride(1), do.stride(2), do.stride(3), dh.stride(1), dh.stride(2), K ** -0.5, H=H, T=T, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT)
    return dh, dv2


def chunk_bwd_dqkw_fn(q, k, v_new, w, g, h, du, do, dh, BT):
    B, H, T, K, V = *q.shape, v_new.shape[-1]
    BK = triton.next_power_of_2(K)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    NT = triton.cdiv(T, BT)
    grid = NK, NT, B * H
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    dg = torch.zeros(NK, *g.shape, dtype=torch.float32, device=g.device)
    chunk_gated_delta_rule_bwd_kernel_dqkw[grid](q, k, v_new, w, g, h, do, dh, dq, dk, du, dw, dg, q.stride(1), q.stride(2), q.stride(3), v_new.stride(1), v_new.stride(2), v_new.stride(3), dh.stride(1), dh.stride(2), scale=K ** -0.5, B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT)
    dg = dg.sum(0)
    return dq, dk, dw, dg


def chunk_fwd_h_fn(k, w, u, g, BT, initial_state, final_state, state_in_fp32=False):
    B, H, T, K, V = *k.shape, u.shape[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, 'current kernel does not support head dimension larger than 256.'
    BV = 16 if BK > 128 else 32
    BV = 64 if BK <= 64 else BV
    BC = 16 if BK > 128 else 32
    BC = 64 if BK <= 64 else BC
    BC = min(BT, BC)
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    h = k.new_empty(B, H, NT * K, V)
    if state_in_fp32:
        h = h.float()
    grid = NK, NV, B * H
    v_new = torch.empty_like(u)
    chunk_gated_delta_rule_fwd_kernel_h[grid](k, u, w, v_new, g, h, initial_state, final_state, k.stride(1), k.stride(2), k.stride(3), u.stride(1), u.stride(2), u.stride(3), h.stride(1), h.stride(2), H=H, T=T, K=K, V=V, BT=BT, BC=BC, BK=BK, BV=BV, NT=NT, USE_INITIAL_STATE=initial_state is not None, STORE_FINAL_STATE=final_state is not None)
    return h, v_new


def chunk_fwd_o_fn(q, k, v_new, g, h, BT):
    B, H, T, K, V = *q.shape, v_new.shape[-1]
    BK = triton.next_power_of_2(K)
    o = torch.empty_like(v_new)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(K), 64)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = NV, NT, B * H
    chunk_linear_attn_fwd_kernel_o[grid](q, k, v_new, g, h, o, q.stride(1), q.stride(2), q.stride(3), v_new.stride(1), v_new.stride(2), v_new.stride(3), h.stride(1), h.stride(2), scale=K ** -0.5, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV)
    return o


def fwd_prepare_du(q, k, g, do, BT):
    dv = torch.empty_like(do)
    B, H, T, K, V = *k.shape, do.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    fwd_prepare_du_kernel[NT, B * H](q, k, g, do, dv, k.stride(1), k.stride(2), k.stride(3), do.stride(1), do.stride(2), do.stride(3), T, K, V, K ** -0.5, BT, BK, BV)
    return dv


def fwd_prepare_wy_repr(k, v, beta, g, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    u = torch.empty_like(v)
    w = torch.empty_like(k)
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    A_w = torch.empty(B, H, T, BT, device=k.device, dtype=torch.float32)
    A_w_original = torch.zeros(B, H, T, BT, device=k.device, dtype=torch.float32)
    A_u = torch.empty(B, H, T, BT, device=k.device, dtype=torch.float32)
    A_u_original = torch.zeros(B, H, T, BT, device=k.device, dtype=torch.float32)
    fwd_prepare_wy_repr_kernel_w[NT, B * H](k, beta, w, A_w, A_w_original, k.stride(1), k.stride(2), k.stride(3), v.stride(1), v.stride(2), v.stride(3), T, K, V, BT, BK, BV)
    fwd_prepare_wy_repr_kernel_u[NT, B * H](k, v, beta, g, u, A_u, A_u_original, k.stride(1), k.stride(2), k.stride(3), v.stride(1), v.stride(2), v.stride(3), T, K, V, BT, BK, BV)
    return w, u, A_w, A_u, A_w_original, A_u_original


def fwd_recompute_w_u(k, v, beta, A_w, A_u, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    u = torch.empty_like(v)
    w = torch.empty_like(k)
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    fwd_recompute_w_u_kernel[NT, B * H](k, v, beta, w, u, A_w, A_u, k.stride(1), k.stride(2), k.stride(3), v.stride(1), v.stride(2), v.stride(3), T, K, V, BT, BK, BV)
    return w, u


def chunk_gated_delta_rule(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', beta: 'torch.Tensor', g: 'torch.Tensor', BT: 'int'=64, initial_state: 'torch.Tensor'=None, output_final_state: 'bool'=False):
    assert q.dtype == k.dtype == v.dtype
    L = q.shape[-2]
    if L % BT != 0:
        q, k, v, beta, g = map(lambda x: F.pad(x, (0, 0, 0, BT - L % BT)), [q, k, v, beta.unsqueeze(-1), g.unsqueeze(-1)])
    g = g.squeeze(-1)
    beta = beta.squeeze(-1)
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkGatedDeltaRuleFunction.apply(q, k, v, beta, g, BT, initial_state, output_final_state)
    return o[:, :, :L, :], final_state


KVCache = Tuple[torch.Tensor, torch.Tensor]


def apply_rotary(x: 'torch.Tensor', cos: 'torch.Tensor', sin: 'torch.Tensor', seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None, interleaved=False, inplace=False, conjugate=False) ->torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, 'If cu_seqlens is passed in, then max_seqlen must be passed'
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, 'rotary_dim must be <= headdim'
    assert seqlen_ro >= seqlen, 'seqlen_ro must be >= seqlen'
    assert cos.dtype == sin.dtype, f'cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}'
    assert x.dtype == cos.dtype, f'Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}'
    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    BLOCK_K = 32 if rotary_dim <= 32 else 64 if rotary_dim <= 64 else 128 if rotary_dim <= 128 else 256

    def grid(META):
        return triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads
    BLOCK_M = 4 if interleaved else 8 if rotary_dim <= 64 else 4
    with torch.device(x.device.index):
        rotary_kernel[grid](output, x, cos, sin, cu_seqlens, seqlen_offsets, seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128, output.stride(0) if not is_varlen else 0, output.stride(-3), output.stride(-2), output.stride(-1), x.stride(0) if not is_varlen else 0, x.stride(-3), x.stride(-2), x.stride(-1), BLOCK_K, isinstance(seqlen_offsets, torch.Tensor), is_varlen, interleaved, conjugate, BLOCK_M)
    return output


class ApplyRotaryEmb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False, seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None):
        out = apply_rotary(x, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=interleaved, inplace=inplace)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(do, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=ctx.max_seqlen, interleaved=ctx.interleaved, inplace=ctx.inplace, conjugate=True)
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets: 'Union[int, torch.Tensor]'=0, cu_seqlens: 'Optional[torch.Tensor]'=None, max_seqlen: 'Optional[int]'=None):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen)


apply_rotary_emb_func = apply_rotary_emb


class CausalSelfAttention(nn.Module):

    def __init__(self, config: 'Config', layer_idx: 'int', n_embd: 'int', head_size=None) ->None:
        super().__init__()
        self.local = layer_idx % config.full_per_layer < config.full_per_layer - 1
        if head_size is not None:
            self.head_size = head_size
            self.n_head = n_embd // head_size
            self.n_query_groups = self.n_head
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups
        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.config = config
        self.sc = config.sc_attn
        if self.sc:
            self.q_dim = self.n_head * self.head_size
            self.kv_dim = self.n_query_groups * self.head_size
            d_conv = 4
            self.q_conv1d = nn.Conv1d(in_channels=self.q_dim, out_channels=self.q_dim, bias=False, kernel_size=d_conv, groups=self.q_dim, padding=d_conv - 1)
            self.k_conv1d = nn.Conv1d(in_channels=self.kv_dim, out_channels=self.kv_dim, bias=False, kernel_size=d_conv, groups=self.kv_dim, padding=d_conv - 1)
            self.v_conv1d = nn.Conv1d(in_channels=self.kv_dim, out_channels=self.kv_dim, bias=False, kernel_size=d_conv, groups=self.kv_dim, padding=d_conv - 1)

    def forward(self, x: 'torch.Tensor', rope: 'RoPECache', max_seq_length: 'int', mask: 'Optional[torch.Tensor]'=None, input_pos: 'Optional[torch.Tensor]'=None, kv_cache: 'Optional[KVCache]'=None) ->Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()
        qkv = self.attn(x)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, -1)
        k = k.reshape(B, T, -1)
        v = v.reshape(B, T, -1)
        if self.sc:
            q = causal_conv1d_fn(x=q.transpose(-1, -2), weight=rearrange(self.q_conv1d.weight, 'd 1 w -> d w'), bias=self.q_conv1d.bias, activation='silu').transpose(-1, -2)
            k = causal_conv1d_fn(x=k.transpose(-1, -2), weight=rearrange(self.k_conv1d.weight, 'd 1 w -> d w'), bias=self.k_conv1d.bias, activation='silu').transpose(-1, -2)
            v = causal_conv1d_fn(x=v.transpose(-1, -2), weight=rearrange(self.v_conv1d.weight, 'd 1 w -> d w'), bias=self.v_conv1d.bias, activation='silu').transpose(-1, -2)
        q = q.reshape(B, T, -1, self.head_size)
        k = k.reshape(B, T, -1, self.head_size)
        v = v.reshape(B, T, -1, self.head_size)
        if not self.config.nope:
            cos, sin = rope
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k, cache_v
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)
            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v
        y = self.scaled_dot_product_attention(q, k, v, mask=mask)
        y = y.reshape(B, T, -1)
        y = self.proj(y)
        return y, kv_cache

    def scaled_dot_product_attention(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None):
        scale = 1.0 / math.sqrt(self.head_size)
        if FlashAttention2Available and mask is None and q.device.type == 'cuda' and q.dtype in (torch.float16, torch.bfloat16):
            if self.local and self.config.local_window > -1:
                win_tuple = self.config.local_window - 1, 0
            else:
                win_tuple = -1, -1
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True, window_size=win_tuple)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None)
        return y.transpose(1, 2)


class Block(nn.Module):

    def __init__(self, config: 'Config', layer_idx: 'int') ->None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.use_gated_deltanet = layer_idx % config.gated_delta_per_layer == 0 if config.gated_delta_per_layer > 0 else False
        if self.use_gated_deltanet:
            self.attn = GatedDeltaNet(hidden_size=config.n_embd)
        else:
            self.attn = CausalSelfAttention(config, n_embd=config.n_embd, layer_idx=layer_idx)
        if not config.shared_attention_norm and config.mlp and not config.parallel_residual:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if config.mlp:
            self.mlp = config.mlp_class(config)
        self.config = config

    def forward(self, x: 'torch.Tensor', rope: 'RoPECache', max_seq_length: 'int', mask: 'Optional[torch.Tensor]'=None, input_pos: 'Optional[torch.Tensor]'=None, kv_cache: 'Optional[KVCache]'=None) ->Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)
        if self.use_gated_deltanet:
            h, _, new_kv_cache = self.attn(n_1, attention_mask=mask)
        else:
            h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)
        if self.config.parallel_residual:
            assert self.config.shared_attention_norm
            if self.config.mlp:
                h = h + self.mlp(n_1)
            x = x + h
        else:
            x = x + h
            if self.config.mlp:
                n_2 = self.norm_2(x)
                h = self.mlp(n_2)
                x = x + h
        return x, new_kv_cache


def _dropout_add_layer_norm_backward(dz, dx, x, x0, dmask, mu, rsigma, gamma, rowscale, colscale, dropout_p, has_residual, is_rms_norm=False):
    """Assume that arguments are contiguous and aligned to 16 bytes
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + residual was not returned in the fwd).
    x0 must not be None if we have colscale.
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    if colscale is not None:
        assert x0 is not None, 'x0 is required to compute the gradient of colscale'
    dx0mat, dresidualmat, dgamma, dbeta, _, _, *rest = dropout_layer_norm.dropout_add_ln_bwd(dzmat, dxmat, xmat, x0mat, dmask, mu, rsigma, gamma, rowscale, colscale, None, None, dropout_p, 1.0, 0, has_residual, is_rms_norm)
    if colscale is None:
        return dx0mat, dresidualmat, dgamma, dbeta
    else:
        dcolscale = rest[0]
        return dx0mat, dresidualmat, dgamma, dbeta, dcolscale


def _dropout_add_layer_norm_forward(x0, residual, gamma, beta, rowscale, colscale, dropout_p, epsilon, residual_in_fp32=False, is_rms_norm=False):
    """Assume that arguments are contiguous and aligned to 16 bytes"""
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(x0mat, residualmat, gamma, beta, rowscale, colscale, None, None, dropout_p, epsilon, 1.0, 0, None, residual_in_fp32, is_rms_norm)
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def maybe_align(x, alignment_in_bytes=16):
    """Assume that x already has last dim divisible by alignment_in_bytes"""
    return x if x.data_ptr() % alignment_in_bytes == 0 else x.clone()


class DropoutAddLayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, residual, gamma, beta, rowscale, colscale, dropout_p, epsilon, residual_in_fp32=False, prenorm=False, is_rms_norm=False, return_dmask=False):
        x0 = maybe_align(x0.contiguous(), 16)
        residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
        gamma = maybe_align(gamma.contiguous(), 16)
        beta = maybe_align(beta.contiguous(), 16) if beta is not None else None
        rowscale = maybe_align(rowscale.contiguous(), 16) if rowscale is not None else None
        colscale = maybe_align(colscale.contiguous(), 16) if colscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(x0, residual, gamma, beta, rowscale, colscale, dropout_p, epsilon, residual_in_fp32, is_rms_norm)
        x0_saved = x0 if colscale is not None else None
        ctx.save_for_backward(xmat.view(x0.shape), x0_saved, dmask, gamma, mu, rsigma, rowscale, colscale)
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta is not None
        if not return_dmask:
            return zmat.view(x0.shape) if not prenorm else (zmat.view(x0.shape), xmat.view(x0.shape))
        else:
            dmask = dmask.view(x0.shape) if dropout_p > 0.0 else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            ctx.mark_non_differentiable(dmask)
            return (zmat.view(x0.shape), dmask) if not prenorm else (zmat.view(x0.shape), xmat.view(x0.shape), dmask)

    @staticmethod
    def backward(ctx, dz, *args):
        dz = maybe_align(dz.contiguous(), 16)
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        x, x0, dmask, gamma, mu, rsigma, rowscale, colscale = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dresidualmat, dgamma, dbeta, *rest = _dropout_add_layer_norm_backward(dz, dx, x, x0, dmask, mu, rsigma, gamma, rowscale, colscale, dropout_p, has_residual, ctx.is_rms_norm)
        dx0 = dx0mat.view(x.shape)
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return dx0, dresidual, dgamma, dbeta if ctx.has_beta else None, None, dcolscale, None, None, None, None, None, None


def rms_norm(x, weight, epsilon):
    return DropoutAddLayerNormFn.apply(x, None, weight, None, None, None, 0.0, epsilon, False, False, True)


class FusedRMSNorm(torch.nn.Module):

    def __init__(self, size: 'int', dim: 'int'=-1, eps: 'float'=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.dim = dim
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def find_multiple(n: 'int', k: 'int') ->int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - n % k


configs = []


name_to_config = {config['name']: config for config in configs}


class LLaMAMLP(nn.Module):

    def __init__(self, config: 'Config') ->None:
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd, config.intermediate_size, bias=config.bias, _pack_weights=False)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.swiglu(x)
        return x


RoPECache = Tuple[torch.Tensor, torch.Tensor]


class MBlock(nn.Module):

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, 'RMSNorm import fails'
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), 'Only LayerNorm and RMSNorm are supported for fused_add_norm'

    def forward(self, hidden_states: 'Tensor', residual: 'Optional[Tensor]'=None, inference_params=None):
        """Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = hidden_states + residual if residual is not None else hidden_states
            hidden_states = self.norm(residual)
            if self.residual_in_fp32:
                residual = residual
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(hidden_states, self.norm.weight, self.norm.bias, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=False, residual_in_fp32=False, fused_add_norm=False, layer_idx=None, device=None, dtype=None):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = MBlock(d_model, mixer_cls, norm_cls=norm_cls, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


def dropout_add_layer_norm(x0, residual, weight, bias, dropout_p, epsilon, rowscale=None, layerscale=None, prenorm=False, residual_in_fp32=False, return_dropout_mask=False):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormFn.apply(x0, residual, weight, bias, rowscale, layerscale, dropout_p, epsilon, residual_in_fp32, prenorm, False, return_dropout_mask)


class DropoutAddLayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, prenorm=False, p=0.0, eps=1e-05, residual_in_fp32=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x0, residual=None):
        return dropout_add_layer_norm(x0, residual, self.weight, self.bias, self.p if self.training else 0.0, self.eps, prenorm=self.prenorm, residual_in_fp32=self.residual_in_fp32)


class RotaryEmbedding(torch.nn.Module):
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

    def __init__(self, dim: 'int', base=10000.0, interleaved=False, scale_base=None, pos_idx_in_fp32=True, device=None):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim) if scale_base is not None else None
        self.register_buffer('scale', scale, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)

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

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', seqlen_offset: 'Union[int, torch.Tensor]'=0, max_seqlen: 'Optional[int]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
             else it's just q of shape (batch, seqlen, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        seqlen = q.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=q.device, dtype=q.dtype)
        if self.scale is None:
            q = apply_rotary_emb_func(q, self._cos_cached, self._sin_cached, interleaved=self.interleaved, seqlen_offsets=seqlen_offset)
            k = apply_rotary_emb_func(k, self._cos_cached, self._sin_cached, interleaved=self.interleaved, seqlen_offsets=seqlen_offset)
        else:
            q = apply_rotary_emb_func(q, self._cos_cached, self._sin_cached, interleaved=self.interleaved, seqlen_offsets=seqlen_offset)
            k = apply_rotary_emb_func(k, self._cos_k_cached, self._sin_k_cached, interleaved=self.interleaved, seqlen_offsets=seqlen_offset)
        return q, k


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (RMSNorm,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NVlabs_GatedDeltaNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

