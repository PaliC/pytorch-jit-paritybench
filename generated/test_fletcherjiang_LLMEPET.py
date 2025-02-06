import sys
_module = sys.modules[__name__]
del sys
llm_epet = _module
attention = _module
config = _module
crossattention = _module
inference = _module
llama = _module
matcher = _module
misc = _module
model = _module
position_encoding = _module
postprocessing_llm = _module
rmsnorm = _module
span_utils = _module
start_end_dataset = _module
train = _module
transformer = _module
eval = _module
utils = _module
basic_utils = _module
model_utils = _module
temporal_nms = _module
tensor_utils = _module
windows_utils = _module

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


import copy


from typing import Optional


from typing import List


import torch


import torch.nn.functional as F


from torch import nn


from torch import Tensor


import warnings


from typing import Tuple


from torch.nn.modules.linear import Linear


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


from torch.nn import functional as F


import math


from torch._C import _infer_size


from torch._C import _add_docstr


from torch.nn import _reduction as _Reduction


from torch.nn.modules import utils


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn.modules.utils import _list_with_default


from torch.nn import grad


from torch import _VF


from torch._jit_internal import boolean_dispatch


from torch._jit_internal import List


from torch._jit_internal import Optional


from torch._jit_internal import _overload


from torch._jit_internal import Tuple


from torch.nn.functional import linear


from torch.nn.functional import pad


from torch.nn.functional import softmax


from torch.nn.functional import dropout


import time


from torch.nn.functional import relu


import numpy as np


from collections import OrderedDict


from collections import defaultdict


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import logging


from scipy.optimize import linear_sum_assignment


import torch.nn as nn


from torch.utils.data import Dataset


import random


from torch.utils.tensorboard import SummaryWriter


from sklearn.manifold import TSNE


import matplotlib.pyplot as plt


from sklearn.decomposition import PCA


def multi_head_attention_forward(query: 'Tensor', key: 'Tensor', value: 'Tensor', embed_dim_to_check: 'int', num_heads: 'int', in_proj_weight: 'Tensor', in_proj_bias: 'Tensor', bias_k: 'Optional[Tensor]', bias_v: 'Optional[Tensor]', add_zero_attn: 'bool', dropout_p: 'float', out_proj_weight: 'Tensor', out_proj_bias: 'Tensor', training: 'bool'=True, key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, use_separate_proj_weight: 'bool'=False, q_proj_weight: 'Optional[Tensor]'=None, k_proj_weight: 'Optional[Tensor]'=None, v_proj_weight: 'Optional[Tensor]'=None, static_k: 'Optional[Tensor]'=None, static_v: 'Optional[Tensor]'=None, out_dim: 'Optional[Tensor]'=None, num_dummies=3, dummy=True) ->Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias
        if any([(type(t) is not Tensor) for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(multi_head_attention_forward, tens_ops, query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight, q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    head_dim = embed_dim // num_heads
    v_head_dim = out_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    q = query * scaling
    k = key
    v = value
    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, 'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
            attn_mask = attn_mask
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn('Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
        key_padding_mask = key_padding_mask
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert bias_k is None
        assert bias_v is None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == v_head_dim
        v = static_v
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights_d = dropout(attn_output_weights, p=dropout_p, training=training)
    if dummy:
        attn_output = torch.bmm(attn_output_weights_d[:, :, num_dummies:], v[:, num_dummies:, :])
    else:
        attn_output = torch.bmm(attn_output_weights_d, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1,\\dots,head_h)W^O
        \\text{where} head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: 'Optional[torch.Tensor]'
    bias_v: 'Optional[torch.Tensor]'

    def __init__(self, embed_dim, num_heads, dropout=0.0, num_dummies=3, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.num_dummies = num_dummies
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        vdim = vdim if vdim is not None else embed_dim
        self.out_proj = Linear(vdim, vdim)
        self.in_proj_bias = None
        self.in_proj_weight = None
        self.bias_k = self.bias_v = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.out_proj.bias, 0.0)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, dummy=True):
        """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*\\text{num_heads}, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, out_dim=self.vdim, num_dummies=self.num_dummies, dummy=dummy)
        else:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, out_dim=self.vdim, num_dummies=self.num_dummies, dummy=dummy)


class RMSNorm(nn.Module):

    def __init__(self, d, p=-1.0, eps=1e-08, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        stdv = 1.0 / math.sqrt(d / 3)
        self.scale.data.uniform_(-stdv, stdv)
        self.register_parameter('scale', self.scale)
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter('offset', self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


def reshape_for_broadcast(freqs_cis: 'torch.Tensor', x: 'torch.Tensor'):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """ Change the AR interfaces by removing the cache tensors.
        Remove the rotary positional embedding.
    """

    def __init__(self, config):
        super().__init__()
        self.n_local_heads = config['n_heads']
        self.head_dim = config['dim'] // config['n_heads']
        self.wq = nn.Linear(config['dim'], config['n_heads'] * self.head_dim, bias=False)
        self.wv = nn.Linear(config['dim'], config['n_heads'] * self.head_dim, bias=False)
        self.wk = nn.Linear(config['dim'], config['n_heads'] * self.head_dim, bias=False)
        self.wo = nn.Linear(config['n_heads'] * self.head_dim, config['dim'], bias=False)

    def forward(self, x: 'torch.Tensor'):
        """ Attention between the agents and the lanes

        Args:
            x (torch.Tensor): features of the tokens, bsz x node_num x dim
            mask (Torch.Tensor): whether the token is valid, bsz x node_num x node_num
        Return:
            feats (torch.Tensor): features of the tokens, bsz x node_num x dim
        """
        bsz, token_num, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, token_num, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, token_num, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, token_num, self.n_local_heads, self.head_dim)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        queries = xq.transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(bsz, token_num, -1)
        return self.wo(output)

    def forward_llama(self, x: 'torch.Tensor', start_pos: 'int', freqs_cis: 'torch.Tensor', mask: 'Optional[torch.Tensor]'):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        self.cache_k = self.cache_k
        self.cache_v = self.cache_v
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int'):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """ Change the AR interfaces by removing the cache tensors.
        Remove the rotary positional embedding.
    """

    def __init__(self, layer_id: 'int', config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.dim = config['dim']
        self.head_dim = self.dim // self.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(dim=self.dim, hidden_dim=4 * self.dim, multiple_of=config['multiple_of'])
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(self.dim, eps=config['norm_eps'])
        self.ffn_norm = RMSNorm(self.dim, eps=config['norm_eps'])

    def forward(self, x: 'torch.Tensor'):
        h = x + self.attention.forward(self.attention_norm(x))
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

    def forward_llama(self, x: 'torch.Tensor', start_pos: 'int', freqs_cis: 'torch.Tensor', mask: 'Optional[torch.Tensor]'):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LLaMATransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers = config['n_layers']
        self.first_layer = config['first_layer']
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.first_layer, self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))
        self.norm = RMSNorm(config['dim'], eps=config['norm_eps'])
        self.prepare_inputs_for_generation = None

    def forward(self, tokens: 'torch.Tensor'):
        bsz, token_num, hidden_dim = tokens.shape
        h = tokens
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return h.float()

    def custom_load_state_dict(self, checkpoint, tail=False, strict=False):
        if tail:
            for i in range(self.first_layer, self.n_layers):
                layer_checkpoint_keys = [k for k in checkpoint.keys() if f'layers.{i}.' in k]
                layer_checkpoint_keys = [k.replace(f'layers.{i}.', '') for k in layer_checkpoint_keys]
                layer_checkpoint = {k: checkpoint[f'layers.{i}.{k}'] for k in layer_checkpoint_keys}
                self.layers[i - self.first_layer].load_state_dict(layer_checkpoint, strict=strict)
        return

    @torch.inference_mode()
    def forward_llama(self, tokens: 'torch.Tensor', start_pos: 'int'):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        if self.adapter:
            adapter_index = 0
            adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, 4096).unsqueeze(1)
        for layer in self.layers:
            if not self.use_adapter:
                h = layer(h, start_pos, freqs_cis, mask)
            else:
                h = layer(h, start_pos, freqs_cis, mask, adapter[adapter_index])
                adapter_index += 1
        h = self.norm(h)
        output = self.output(h[:, -1, :])
        return output.float()


def temporal_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> temporal_iou(test_spans1, test_spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]
    areas2 = spans2[:, 1] - spans2[:, 0]
    left = torch.max(spans1[:, None, 0], spans2[:, 0])
    right = torch.min(spans1[:, None, 1], spans2[:, 1])
    inter = (right - left).clamp(min=0)
    union = areas1[:, None] + areas2 - inter
    iou = inter / union
    return iou, union


def generalized_temporal_iou(spans1, spans2):
    """
    Generalized IoU from https://giou.stanford.edu/
    Also reference to DETR implementation of generalized_box_iou
    https://github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40

    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor

    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> generalized_temporal_iou(test_spans1, test_spans2)
    tensor([[ 0.6667,  0.2000],
        [-0.2000,  0.5000]])
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)
    left = torch.min(spans1[:, None, 0], spans2[:, 0])
    right = torch.max(spans1[:, None, 1], spans2[:, 1])
    enclosing_area = (right - left).clamp(min=0)
    return iou - (enclosing_area - union) / enclosing_area


def span_cxw_to_xx(cxw_spans):
    """
    Args:
        cxw_spans: tensor, (#windows, 2) or (..., 2), the last dim is a row denoting a window of format (center, width)

    >>> spans = torch.Tensor([[0.5000, 1.0000], [0.3000, 0.2000]])
    >>> span_cxw_to_xx(spans)
    tensor([[0.0000, 1.0000],
        [0.2000, 0.4000]])
    >>> spans = torch.Tensor([[[0.5000, 1.0000], [0.3000, 0.2000]]])
    >>> span_cxw_to_xx(spans)
    tensor([[[0.0000, 1.0000],
        [0.2000, 0.4000]]])
    """
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: 'float'=1, cost_span: 'float'=1, cost_giou: 'float'=1, span_loss_type: 'str'='l1', max_v_l: 'int'=75):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs['pred_spans'].shape[:2]
        targets = targets['span_labels']
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        tgt_spans = torch.cat([v['spans'] for v in targets])
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)
        cost_class = -out_prob[:, tgt_ids]
        if self.span_loss_type == 'l1':
            out_spans = outputs['pred_spans'].flatten(0, 1)
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)
            cost_giou = -generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        else:
            pred_spans = outputs['pred_spans']
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)
            cost_span = -pred_spans[:, 0][:, tgt_spans[:, 0]] - pred_spans[:, 1][:, tgt_spans[:, 1]]
            cost_giou = 0
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v['spans']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def generalized_temporal_iou_(spans1, spans2):
    """
    Generalized IoU from https://giou.stanford.edu/
    Also reference to DETR implementation of generalized_box_iou
    https://github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40

    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor

    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> generalized_temporal_iou(test_spans1, test_spans2)
    tensor([[ 0.6667,  0.2000],
        [-0.2000,  0.5000]])
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    iou, union = temporal_iou(spans1, spans2)
    left = torch.min(spans1[:, None, 0], spans2[:, 0])
    right = torch.max(spans1[:, None, 1], spans2[:, 1])
    enclosing_area = (right - left).clamp(min=0)
    return iou - (enclosing_area - union) / enclosing_area


class HungarianEventMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_span: 'float'=1, cost_giou: 'float'=1, span_loss_type: 'str'='l1', max_v_l: 'int'=75):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_span != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs.shape[:2]
        tgt_spans = torch.cat([v for v in targets])
        out_spans = outputs.flatten(0, 1)
        cost_span = torch.cdist(out_spans, tgt_spans, p=1)
        cost_giou = -generalized_temporal_iou_(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        C = self.cost_span * cost_span + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [nn.Dropout(dropout), nn.Linear(input_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, **kwargs):
        output = src
        intermediate = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if self.return_intermediate:
                intermediate.append(output)
        if self.norm is not None:
            output = self.norm(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    x = x.div(keep_prob) * mask
    return x


class DropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        x = x.permute(1, 0, 2)
        res = drop_path(x, self.drop_prob, self.training)
        return res.permute(1, 0, 2)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    if activation == 'prelu':
        return nn.PReLU()
    if activation == 'selu':
        return F.selu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(dropout)
        self.dropout2 = DropPath(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        pass

    def forward(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a == b:
            res.append(True)
        else:
            res.append(False)
    return res


def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start + len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def inverse_sigmoid(x, eps=0.001):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class LLM_EPET(nn.Module):
    """ LLM EPET. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim, num_queries, input_dropout, aux_loss=False, contrastive_align_loss=False, contrastive_hdim=64, max_v_l=75, span_loss_type='l1', use_txt_pos=False, n_input_proj=2, aud_dim=0, args=None):
        super().__init__()
        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == 'l1' else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.event_span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        nn.init.constant_(self.event_span_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.event_span_embed.layers[-1].bias.data, 0)
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.input_txt_proj = nn.Sequential(*[LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]), LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]), LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]), LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]), LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])][:n_input_proj])
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(args.total_prompts, hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(1, hidden_dim))
        self.moment_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.moment_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))
        self.dummy_rep_token = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        normalize_before = False
        self.sent_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.sent_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))
        self.txt_proj_linear = LinearLayer(txt_dim, hidden_dim, layer_norm=True)
        input_txt_sa_proj = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, 'prelu', normalize_before)
        txtproj_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, args.dummy_layers, txtproj_encoder_norm)
        scls_encoder_layer = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, 'prelu', normalize_before)
        scls_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.scls_encoder = TransformerEncoder(scls_encoder_layer, args.sent_layers, scls_encoder_norm)
        dim = 4096
        self.llama_dim_mapper1 = nn.Linear(hidden_dim, dim, bias=False)
        self.llama_dim_mapper2 = nn.Linear(dim, hidden_dim, bias=False)
        self.token_fc = nn.Linear(max_v_l, self.args.num_dummies)

    def generate_pseudo_event(self, src_vid, src_vid_mask, targets):
        bsz, L_src, _ = src_vid.size()
        norm_vid = src_vid / (src_vid.norm(dim=2, keepdim=True) + 1e-08)
        tsm = torch.bmm(norm_vid, norm_vid.transpose(1, 2))
        mask = torch.tensor([[1.0, 1.0, 0.0, -1.0, -1.0], [1.0, 1.0, 0.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 1.0, 1.0], [-1.0, -1.0, 0.0, 1.0, 1.0]], device=src_vid.device)
        mask_size = mask.size(0)
        mask = mask.view(1, mask_size, mask_size)
        pad_tsm = nn.ZeroPad2d(mask_size // 2)(tsm)
        score = torch.diagonal(F.conv2d(pad_tsm.unsqueeze(1), mask.unsqueeze(1)).squeeze(1), dim1=1, dim2=2)
        tau = score.mean(1).unsqueeze(1).repeat(1, L_src)
        L_vid = torch.count_nonzero(src_vid_mask, 1)
        st_ed = torch.cat([torch.zeros_like(L_vid).unsqueeze(1), L_vid.unsqueeze(1) - 1], dim=-1)
        score[torch.arange(score.size(0)).unsqueeze(1), st_ed] = 100
        score_r = torch.roll(score, 1, -1)
        score_l = torch.roll(score, -1, -1)
        bnds = torch.where((score_r <= score) & (score_l <= score) & (tau <= score), 1.0, 0.0)
        bnd_indices = bnds.nonzero()
        temp = torch.roll(bnd_indices, 1, 0)
        center = (bnd_indices + temp) / 2
        width = bnd_indices - temp
        bnd_spans = torch.cat([center, width[:, 1:]], dim=-1)
        pseudo_event_spans = [(bnd_spans[bnd_spans[:, 0] == i, :][:, 1:] / L_vid[i]) for i in range(bsz)]
        pseudo_event_spans_used = [bnd_spans[bnd_spans[:, 0] == i, :][:, 1:] for i in range(bsz)]
        return pseudo_event_spans, pseudo_event_spans_used

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, src_aud=None, src_aud_mask=None, targets=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if vid is not None:
            _count = [v.count('_') for v in vid]
            if self.args.dset_name == 'hl':
                _position_to_cut = [find_nth(v, '_', _count[i] - 1) for i, v in enumerate(vid)]
                ori_vid = [v[:_position_to_cut[i]] for i, v in enumerate(vid)]
            else:
                ori_vid = [v for v in vid]
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        pos_vid = self.position_embed(src_vid, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        txt_dummy = self.dummy_rep_token.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(src_txt.shape[0], 1, 1)
        src_txt_dummy = torch.cat([txt_dummy, src_txt], dim=1)
        mask_txt = torch.tensor([[True] * self.args.num_dummies]).repeat(src_txt_mask.shape[0], 1)
        src_txt_mask_dummy = torch.cat([mask_txt, src_txt_mask], dim=1)
        pos_dummy = self.dummy_rep_pos.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(pos_txt.shape[0], 1, 1)
        pos_txt_dummy = torch.cat([pos_dummy, pos_txt], dim=1)
        src_txt_dummy = src_txt_dummy.permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)
        memory = self.txtproj_encoder(src_txt_dummy, src_key_padding_mask=~src_txt_mask_dummy.bool(), pos=pos_txt_dummy)
        dummy_token = memory[:self.args.num_dummies].permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)
        src_txt_dummy = torch.cat([dummy_token, src_txt], dim=1)
        mask_txt_dummy = torch.tensor([[True] * self.args.num_dummies]).repeat(src_txt_mask.shape[0], 1)
        src_txt_mask_dummy = torch.cat([mask_txt_dummy, src_txt_mask], dim=1)
        src = torch.cat([src_vid, src_txt_dummy], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask_dummy], dim=1).bool()
        pos = torch.cat([pos_vid, pos_txt_dummy], dim=1)
        smask_ = torch.tensor([[True]]).repeat(src_txt_mask.shape[0], 1)
        smask = torch.cat([smask_, src_txt_mask.bool()], dim=1)
        ssrc_ = self.sent_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_txt.shape[0], 1, 1)
        ssrc = torch.cat([ssrc_, src_txt], dim=1)
        spos_ = self.sent_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_txt.shape[0], 1, 1)
        spos = torch.cat([spos_, pos_txt], dim=1)
        smaskd = torch.cat([smask_, mask_txt_dummy.bool()], dim=1)
        ssrcd = torch.cat([ssrc_, dummy_token], dim=1)
        sposd = torch.cat([spos_, pos_dummy], dim=1)
        if targets is not None:
            mmask_ = torch.tensor([[True]]).repeat(src_vid_mask.shape[0], 1)
            mmask = torch.cat([mmask_, src_vid_mask.bool()], dim=1)
            moment_mask_ = torch.clamp(targets['relevant_clips'], 0, 1).bool()
            moment_mask = torch.cat([mmask_, moment_mask_], dim=1)
            mmask = mmask * moment_mask
            msrc_ = self.moment_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_vid.shape[0], 1, 1)
            msrc = torch.cat([msrc_, src_vid], dim=1)
            mpos_ = self.moment_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_vid.shape[0], 1, 1)
            mpos = torch.cat([mpos_, pos_vid], dim=1)
            nmmask_ = torch.tensor([[True]]).repeat(src_vid_mask.shape[0], 1)
            nmmask = torch.cat([nmmask_, src_vid_mask.bool()], dim=1)
            nmoment_mask_ = ~torch.clamp(targets['relevant_clips'], 0, 1).bool()
            nmoment_mask = torch.cat([nmmask_, nmoment_mask_], dim=1)
            nmmask = nmmask * nmoment_mask
            nmsrc_ = self.moment_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_vid.shape[0], 1, 1)
            nmsrc = torch.cat([nmsrc_, src_vid], dim=1)
            nmpos_ = self.moment_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_vid.shape[0], 1, 1)
            nmpos = torch.cat([nmpos_, pos_vid], dim=1)
        else:
            moment_mask_ = None
        vidsrc_ = torch.zeros((len(src_vid), 1, self.hidden_dim))
        for i in range(len(src_vid)):
            vidsrc_[i] = src_vid[i][:src_vid_mask.sum(1)[i].long()].mean(0).clone().detach()
        video_length = src_vid.shape[1]
        if targets is not None:
            ssrc = ssrc.permute(1, 0, 2)
            spos = spos.permute(1, 0, 2)
            smemory = self.scls_encoder(ssrc, src_key_padding_mask=~smask, pos=spos)
            sentence_txt, smemory_words = smemory[0], smemory[1:]
            ssrcd = ssrcd.permute(1, 0, 2)
            sposd = sposd.permute(1, 0, 2)
            smemoryd = self.scls_encoder(ssrcd, src_key_padding_mask=~smaskd, pos=sposd)
            sentence_dummy, smemory_words_dummy = smemoryd[0], smemoryd[1:]
            txt_dummy_proj = torch.cat([smemory_words_dummy, smemory_words], dim=0)
            hs, reference, memory, memory_global, attn_weights, memory_moment, nmmemory_moment, mmemory_frames, nmmemory_frames = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length, moment_idx=targets['relevant_clips'], msrc=msrc, mpos=mpos, mmask=~mmask, nmsrc=nmsrc, nmpos=nmpos, nmmask=~nmmask, ctxtoken=vidsrc_, gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask.sum(1).long())
            moment2txt_similarity = torch.matmul(mmemory_frames.permute(1, 0, 2), txt_dummy_proj.permute(1, 2, 0))
            nmoment2txt_similarity = torch.matmul(nmmemory_frames.permute(1, 0, 2), txt_dummy_proj.permute(1, 2, 0))
        else:
            sentence_dummy, sentence_txt, moment2txt_similarity, nmoment2txt_similarity = None, None, None, None
            hs, reference, memory, memory_global, attn_weights, memory_moment, nmmemory_moment, mmemory_frames, nmmemory_frames = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length, ctxtoken=vidsrc_, gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask.sum(1).long())
        pseudo_event_spans, pseudo_event_spans_used = self.generate_pseudo_event(src_vid, src_vid_mask, targets)
        event_tmp = self.event_span_embed(hs[0])
        event_outputs_coord = event_tmp.sigmoid()
        outputs_class = self.class_embed(hs)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == 'l1':
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
        out['pseudo_event_spans'] = pseudo_event_spans
        out['pred_event_spans'] = event_outputs_coord
        out['pseudo_event_spans_used'] = pseudo_event_spans_used
        out['pos'] = pos
        txt_mem = memory[:, src_vid.shape[1]:]
        vid_mem = memory[:, :src_vid.shape[1]]
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(proj_queries=proj_queries[-1], proj_txt_mem=proj_txt_mem, proj_vid_mem=proj_vid_mem))
        if vid is not None:
            neg_vid = ori_vid[1:] + ori_vid[:1]
            real_neg_mask = torch.Tensor(element_wise_list_equal(ori_vid, neg_vid))
            real_neg_mask = real_neg_mask == False
            if real_neg_mask.sum() != 0:
                src_txt_dummy_neg = torch.cat([src_txt_dummy[1:], src_txt_dummy[0:1]], dim=0)
                src_txt_mask_dummy_neg = torch.cat([src_txt_mask_dummy[1:], src_txt_mask_dummy[0:1]], dim=0)
                src_dummy_neg = torch.cat([src_vid, src_txt_dummy_neg], dim=1)
                mask_dummy_neg = torch.cat([src_vid_mask, src_txt_mask_dummy_neg], dim=1).bool()
                pos_neg = pos.clone()
                mask_dummy_neg = mask_dummy_neg[real_neg_mask]
                src_dummy_neg = src_dummy_neg[real_neg_mask]
                pos_neg = pos_neg[real_neg_mask]
                src_txt_mask_dummy_neg = src_txt_mask_dummy_neg[real_neg_mask]
                _, _, memory_neg, memory_global_neg, attn_weights_neg, _, _, _, _ = self.transformer(src_dummy_neg, ~mask_dummy_neg, self.query_embed.weight, pos_neg, video_length=video_length, ctxtoken=vidsrc_[real_neg_mask], gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask[real_neg_mask].sum(1).long())
                vid_mem_neg = memory_neg[:, :src_vid.shape[1]]
                out['saliency_scores_neg'] = torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim)
                out['src_txt_mask_neg'] = src_txt_mask_dummy_neg
                out['t2vattnvalues_neg'] = (attn_weights_neg[:, :, self.args.num_dummies:] * src_txt_mask_dummy_neg[:, self.args.num_dummies:].unsqueeze(1).repeat(1, video_length, 1)).sum(2)
                out['t2vattnvalues_neg'] = torch.clamp(out['t2vattnvalues_neg'], 0, 1)
            else:
                out['saliency_scores_neg'] = None
                out['t2vattnvalues_neg'] = None
            out['real_neg_mask'] = real_neg_mask
        else:
            out['saliency_scores_neg'] = None
            out['t2vattnvalues_neg'] = None
            out['real_neg_mask'] = None
        out['saliency_scores'] = torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim)
        out['memory_moment'] = memory_moment
        out['nmmemory_moment'] = nmmemory_moment
        out['sentence_txt'] = sentence_txt
        out['sentence_dummy'] = sentence_dummy
        out['moment2txt_similarity'] = moment2txt_similarity
        out['nmoment2txt_similarity'] = nmoment2txt_similarity
        out['cate_attn_weights'] = attn_weights
        out['moment_mask'] = moment_mask_
        out['txt_mask'] = src_txt_mask_dummy
        out['t2vattnvalues'] = (attn_weights[:, :, self.args.num_dummies:] * src_txt_mask.unsqueeze(1).repeat(1, video_length, 1)).sum(2)
        out['t2vattnvalues'] = torch.clamp(out['t2vattnvalues'], 0, 1)
        out['dummy_tokens'] = dummy_token
        out['global_rep_tokens'] = self.global_rep_token
        if targets is not None:
            out['src_vid'] = mmemory_frames.permute(1, 0, 2) * moment_mask_.unsqueeze(2) + nmmemory_frames.permute(1, 0, 2) * (~moment_mask_.unsqueeze(2).bool()).float()
        else:
            out['src_vid'] = None
        out['video_mask'] = src_vid_mask
        if self.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: (#items, #classes)
    target: int,
    """
    maxk = max(topk)
    num_items = output.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / num_items))
    return res


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l, saliency_margin=1, event_matcher=None, use_matcher=True, args=None):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.args = args
        self.matcher = matcher
        self.event_matcher = event_matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.use_matcher = use_matcher
        self.criterion = torch.nn.CrossEntropyLoss()
        self.l2_criterion = torch.nn.MSELoss()
        self.kld_criterion = torch.nn.KLDivLoss(reduction='none')
        self.bce_criterion = nn.BCELoss(reduction='none')

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets['span_labels']
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if self.span_loss_type == 'l1':
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])
        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = self.foreground_label
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')
        losses = {'loss_label': loss_ce.mean()}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if 'saliency_pos_labels' not in targets:
            return {'loss_saliency': 0}
        if outputs['saliency_scores_neg'] is not None:
            vid_token_mask = outputs['video_mask']
            real_neg_mask = outputs['real_neg_mask']
            saliency_scores_neg = outputs['saliency_scores_neg'].clone()
            loss_neg_pair = (-torch.log(1.0 - torch.sigmoid(saliency_scores_neg)) * vid_token_mask[real_neg_mask]).sum(dim=1).mean()
            saliency_scores = outputs['saliency_scores'].clone()
            saliency_contrast_label = targets['saliency_all_labels']
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat([saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (1.0 - realneg_vid_token_mask) * -1000.0
            tau = 0.5
            loss_rank_contrastive = 0.0
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)
                pos_mask = realneg_saliency_contrast_label >= rand_idx
                if torch.sum(pos_mask) == 0:
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1000.0
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)
                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-06)
                loss = -mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12
            false_neg_mask = ~real_neg_mask
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1.0 - falseneg_vid_token_mask) * -1000.0
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1.0 - falseneg_vid_token_mask) * -1000.0
                tau = 0.5
                falseneg_loss_rank_contrastive = 0.0
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)
                    pos_mask = falseneg_saliency_contrast_label >= rand_idx
                    if torch.sum(pos_mask) == 0:
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1000.0
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)
                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-06)
                    loss = -mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive
            saliency_scores = outputs['saliency_scores']
            pos_indices = targets['saliency_pos_labels']
            neg_indices = targets['saliency_neg_labels']
            num_pairs = pos_indices.shape[1]
            batch_indices = torch.arange(len(saliency_scores))
            pos_scores = torch.stack([saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack([saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() / (len(pos_scores) * num_pairs) * 2
            if self.args.dset_name in ['youtube_uni']:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair * 0.0
            else:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
            """higher scores for positive clips"""
            vid_token_mask = outputs['video_mask']
            if outputs['t2vattnvalues_neg'] is not None:
                saliency_scores_neg = outputs['t2vattnvalues_neg'].clone()
                loss_neg_pair_attn = (-torch.log(1.0 - saliency_scores_neg) * vid_token_mask[real_neg_mask]).sum(dim=1).mean()
            saliency_scores = outputs['t2vattnvalues'].clone()
            saliency_contrast_label = targets['saliency_all_labels']
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat([saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (1.0 - realneg_vid_token_mask) * -1000.0
            tau = 0.5
            loss_rank_contrastive_attn = 0.0
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)
                pos_mask = realneg_saliency_contrast_label >= rand_idx
                if torch.sum(pos_mask) == 0:
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1000.0
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)
                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-06)
                loss = -mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive_attn = loss_rank_contrastive_attn + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive_attn / 12
            false_neg_mask = ~real_neg_mask
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1.0 - falseneg_vid_token_mask) * -1000.0
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1.0 - falseneg_vid_token_mask) * -1000.0
                tau = 0.5
                falseneg_loss_rank_contrastive = 0.0
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)
                    pos_mask = falseneg_saliency_contrast_label >= rand_idx
                    if torch.sum(pos_mask) == 0:
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1000.0
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)
                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-06)
                    loss = -mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive
            saliency_scores = outputs['t2vattnvalues']
            pos_indices = targets['saliency_pos_labels']
            neg_indices = targets['saliency_neg_labels']
            num_pairs = pos_indices.shape[1]
            batch_indices = torch.arange(len(saliency_scores))
            pos_scores = torch.stack([saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack([saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() / (len(pos_scores) * num_pairs) * 2
            saliency_binary_label = torch.clamp(targets['saliency_all_labels'], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)
            if self.args.dset_name in ['youtube_uni']:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn * 0 + loss_saliency_attn
            else:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn + loss_saliency_attn
            loss_saliency += loss_saliency_attn * self.args.lw_wattn
        else:
            vid_token_mask = outputs['video_mask']
            saliency_scores = outputs['saliency_scores'].clone()
            saliency_contrast_label = targets['saliency_all_labels']
            saliency_scores = vid_token_mask * saliency_scores + (1.0 - vid_token_mask) * -1000.0
            tau = 0.5
            loss_rank_contrastive = 0.0
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)
                pos_mask = saliency_contrast_label >= rand_idx
                if torch.sum(pos_mask) == 0:
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1000.0
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)
                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-06)
                loss = -mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12
            saliency_scores = outputs['saliency_scores']
            pos_indices = targets['saliency_pos_labels']
            neg_indices = targets['saliency_neg_labels']
            num_pairs = pos_indices.shape[1]
            batch_indices = torch.arange(len(saliency_scores))
            pos_scores = torch.stack([saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack([saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() / (len(pos_scores) * num_pairs) * 2
            loss_saliency = loss_saliency + loss_rank_contrastive
            """higher scores for positive clips"""
            vid_token_mask = outputs['video_mask']
            saliency_scores = outputs['t2vattnvalues'].clone()
            saliency_contrast_label = targets['saliency_all_labels']
            saliency_scores = vid_token_mask * saliency_scores + (1.0 - vid_token_mask) * -1000.0
            tau = 0.5
            loss_rank_contrastive = 0.0
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)
                pos_mask = saliency_contrast_label >= rand_idx
                if torch.sum(pos_mask) == 0:
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1000.0
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)
                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-06)
                loss = -mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive / 12
            saliency_scores = outputs['t2vattnvalues']
            pos_indices = targets['saliency_pos_labels']
            neg_indices = targets['saliency_neg_labels']
            num_pairs = pos_indices.shape[1]
            batch_indices = torch.arange(len(saliency_scores))
            pos_scores = torch.stack([saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack([saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() / (len(pos_scores) * num_pairs) * 2
            saliency_binary_label = torch.clamp(targets['saliency_all_labels'], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)
            loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_saliency_attn
            loss_saliency += loss_saliency_attn * self.args.lw_wattn
        return {'loss_saliency': loss_saliency}

    def loss_contrastive_moment_sentence(self, outputs, targets, indices, log=True):
        if outputs['memory_moment'] is not None:
            moment_token = outputs['memory_moment']
            nmmemory_moment = outputs['nmmemory_moment']
            sentence_token = outputs['sentence_txt'].squeeze(1)
            sentence_dummy = outputs['sentence_dummy'].squeeze(1)
            moment_logits = F.normalize(moment_token, dim=1)
            nmoment_logits = F.normalize(nmmemory_moment, dim=1)
            sentence_logits = F.normalize(sentence_token, dim=1)
            dummy_logits = F.normalize(sentence_dummy, dim=1)
            similarity_matrix = torch.matmul(moment_logits, sentence_logits.T)
            nsimilarity_matrix = torch.matmul(nmoment_logits, sentence_logits.T)
            similarity_matrix = torch.cat([similarity_matrix, nsimilarity_matrix], dim=1)
            labels = torch.eye(similarity_matrix.shape[0])
            nlabels = torch.zeros_like(nsimilarity_matrix)
            labels = torch.cat([labels, nlabels], dim=1).max(dim=1)[1]
            loss_ms_align = self.criterion(similarity_matrix, labels)
            dummy_similarity_matrix = torch.matmul(moment_logits, dummy_logits.T)
            dummy_nsimilarity_matrix = torch.matmul(nmoment_logits, dummy_logits.T)
            dummy_similarity_matrix = torch.cat([dummy_similarity_matrix, dummy_nsimilarity_matrix], dim=1)
            dummy_labels = (~torch.eye(similarity_matrix.shape[0]).bool()).float()
            dummy_nlabels = torch.ones_like(nsimilarity_matrix)
            dummy_labels = torch.cat([dummy_labels, dummy_nlabels], dim=1).max(dim=1)[1]
            dummy_loss_ms_align = self.criterion(dummy_similarity_matrix, dummy_labels)
            loss_ms_align += dummy_loss_ms_align
            video_mask = outputs['video_mask']
            src_vid = outputs['src_vid']
            moment_mask_ = torch.clamp(targets['relevant_clips'], 0, 1)
            momtokcls_pred = torch.matmul(moment_token.unsqueeze(1), src_vid.permute(0, 2, 1))
            momtokcls_label = moment_mask_
            momtokcls_logit = torch.sigmoid(momtokcls_pred)
            loss_ms_align += (self.bce_criterion(momtokcls_logit.reshape(-1), momtokcls_label.reshape(-1)) * video_mask.reshape(-1)).mean()
        else:
            loss_ms_align = 0.0
        return {'loss_ms_align': loss_ms_align}

    def loss_moment2txt_sim_distill(self, outputs, targets, indices, log=True):
        if outputs['moment2txt_similarity'] is not None:
            moment2txt_similarity = outputs['moment2txt_similarity']
            moment_mask = outputs['moment_mask'].int()
            txt_mask = outputs['txt_mask'].unsqueeze(1).repeat(1, outputs['cate_attn_weights'].size(1), 1)
            attn_weights = outputs['cate_attn_weights']
            b, L_vid, L_txt = attn_weights.size()
            loss_distill = self.kld_criterion(torch.log(attn_weights + 1e-06).reshape(b * L_vid, -1), torch.softmax(moment2txt_similarity, dim=-1).clone().detach().reshape(b * L_vid, -1)).mean(1) * moment_mask.reshape(-1)
            loss_distill = loss_distill.sum() / moment_mask.sum()
        else:
            loss_distill = 0.0
        return {'loss_distill': loss_distill}

    def loss_orthogonal_dummy(self, outputs, targets, indices, log=True):
        dummy_tokens = outputs['dummy_tokens']
        if dummy_tokens.size(1) != 1:
            dummy_tokens_norm = dummy_tokens / dummy_tokens.norm(dim=2)[:, :, None]
            dummy_tokens_sim = torch.matmul(dummy_tokens_norm, dummy_tokens_norm.permute(0, 2, 1).detach())
            for i in range(len(dummy_tokens_sim)):
                dummy_tokens_sim[i].fill_diagonal_(0)
            loss_dummy_ortho = dummy_tokens_sim.abs().mean()
        else:
            loss_dummy_ortho = 0.0
        global_tokens = outputs['global_rep_tokens']
        global_tokens_norm = global_tokens / global_tokens.norm(dim=1)[:, None]
        global_tokens_sim = torch.matmul(global_tokens_norm, global_tokens_norm.permute(1, 0).detach())
        for i in range(len(global_tokens_sim)):
            global_tokens_sim.fill_diagonal_(0)
        loss_dummy_ortho += global_tokens_sim.abs().mean()
        return {'loss_orthogonal_dummy': loss_dummy_ortho}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs['proj_txt_mem']
        normalized_img_embed = outputs['proj_queries']
        logits = torch.einsum('bmd,bnd->bmn', normalized_img_embed, normalized_text_embed)
        logits = logits.sum(2) / self.temperature
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)
        pos_term = positive_logits.sum(1)
        num_pos = positive_map.sum(1)
        neg_term = logits.logsumexp(1)
        loss_nce = -pos_term / num_pos + neg_term
        losses = {'loss_contrastive_align': loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs['proj_txt_mem']
        normalized_img_embed = outputs['proj_queries']
        logits = torch.einsum('bmd,bnd->bmn', normalized_img_embed, normalized_text_embed)
        logits = logits.sum(2) / self.temperature
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)
        pos_term = positive_logits.sum(1)
        num_pos = positive_map.sum(1)
        neg_term = logits.logsumexp(1)
        loss_nce = -pos_term / num_pos + neg_term
        losses = {'loss_contrastive_align': loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def loss_event_spans(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_event_spans'][idx]
        tgt_spans = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_event_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
        loss_event_giou = 1 - torch.diag(generalized_temporal_iou_(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        return {'loss_event_span': loss_event_span.mean(), 'loss_event_giou': loss_event_giou.mean()}

    def loss_pos(self, outputs, targets, indices):
        src_pos = outputs['pos']
        src_event = outputs['pseudo_event_spans_used']
        total_loss = 0
        for video_idx in range(len(src_pos)):
            video_loss = 0
            for span in src_event[video_idx]:
                center, width = span
                start_idx = int(center - width / 2)
                end_idx = int(center + width / 2)
                center, width = span.int()
                target_values = src_pos[video_idx, center, :].unsqueeze(0)
                if end_idx - start_idx < 0:
                    video_loss += torch.exp(step_loss)
                    continue
                target_values = target_values.repeat(end_idx - start_idx, 1)
                before_center = src_pos[video_idx][start_idx:center]
                after_center = src_pos[video_idx][center + 1:end_idx + 1]
                time_step_values = torch.cat((before_center, after_center), dim=0)
                step_loss = abs(time_step_values - target_values).mean()
                video_loss += torch.exp(step_loss)
            total_loss += video_loss
        return {'loss_pos': total_loss / len(src_pos)}

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {'spans': self.loss_spans, 'labels': self.loss_labels, 'contrastive_align': self.loss_contrastive_align, 'saliency': self.loss_saliency, 'ms_align': self.loss_contrastive_moment_sentence, 'distill': self.loss_moment2txt_sim_distill, 'orthogonal_dummy': self.loss_orthogonal_dummy, 'event_spans': self.loss_event_spans, 'pos': self.loss_pos}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            event_indices = self.event_matcher(outputs['pred_event_spans'], outputs['pseudo_event_spans'])
            losses_target = self.losses
        else:
            indices = None
            losses_target = ['saliency']
        losses = {}
        for loss in losses_target:
            if loss == 'event_spans':
                indices_in = event_indices
                targets_in = outputs['pseudo_event_spans']
            else:
                indices_in = indices
                targets_in = targets
            losses.update(self.get_loss(loss, outputs, targets_in, indices_in))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ['saliency', 'ms_align', 'distill', 'orthogonal_dummy']
                for loss in losses_target:
                    if 'saliency' == loss:
                        continue
                    if 'ms_align' == loss:
                        continue
                    if 'distill' == loss:
                        continue
                    if 'orthogonal_dummy' == loss:
                        continue
                    if 'event_spans' == loss:
                        continue
                    if 'pos' == loss:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        """
        Args:
            input_feat: (N, L, D)
        """
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid

        Returns:

        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = torch.div(dim_t, 2, rounding_mode='trunc')
        dim_t = self.temperature ** (2 * dim_t / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos_x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, mask):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1), y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, num_dummies=3):
        super().__init__()
        self.self_attn = cateattention(d_model, nhead, dropout=dropout, num_dummies=num_dummies)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(dropout)
        self.dropout2 = DropPath(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, video_length=None, dummy=True):
        assert video_length is not None
        pos_src = self.with_pos_embed(src, pos)
        q, k, v = pos_src[:video_length], pos_src[video_length:], src[video_length:]
        qmask, kmask = src_key_padding_mask[:, :video_length].unsqueeze(2), src_key_padding_mask[:, video_length:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)
        src2, attn_weights = self.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask[:, video_length:], dummy=dummy)
        src2 = src[:video_length] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src = torch.cat([src2, src[video_length:]])
        return src, attn_weights

    def forward_pre(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, dummy=True):
        pass

    def forward(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, dummy=True, **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, dummy=dummy)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, dummy=dummy, **kwargs)


class TransformerCATEEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, dummy=True, **kwargs):
        output = src
        intermediate = []
        attn_weights = None
        for i, layer in enumerate(self.layers):
            output, attn_weight = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos, dummy=dummy, **kwargs)
            if attn_weights is None:
                attn_weights = attn_weight
            else:
                attn_weights = attn_weights + attn_weight
            if self.return_intermediate:
                intermediate.append(output)
        attn_weights /= self.num_layers
        if self.norm is not None:
            output = self.norm(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, attn_weights


def gen_sineembed_for_position(pos_tensor, d_model):
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise', modulate_t_attn=False, bbox_embed_diff_each_layer=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(query_scale_type))
        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 2, 3) for i in range(num_layers)])
        else:
            self.bbox_embed = MLP(d_model, d_model, 2, 3)
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if modulate_t_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, refpoints_unsigmoid: 'Optional[Tensor]'=None):
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model)
            query_pos = self.ref_point_head(query_sine_embed)
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            query_sine_embed = query_sine_embed * pos_transformation
            if self.modulate_t_attn:
                reft_cond = self.ref_anchor_head(output).sigmoid()
                query_sine_embed *= (reft_cond[..., 0] / obj_center[..., 1]).unsqueeze(-1)
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed, is_first=layer_id == 0)
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [torch.stack(intermediate).transpose(1, 2), torch.stack(ref_points).transpose(1, 2)]
            else:
                return [torch.stack(intermediate).transpose(1, 2), reference_points.unsqueeze(0).transpose(1, 2)]
        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, keep_query_pos=False, rm_self_attn_decoder=False):
        super().__init__()
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = DropPath(dropout)
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = DropPath(dropout)
        self.dropout3 = DropPath(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None, query_sine_embed=None, is_first=False):
        if not self.rm_self_attn_decoder:
            q_content = self.sa_qcontent_proj(tgt)
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)
            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape
            q = q_content + q_pos
            k = k_content + k_pos
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape
        k_pos = self.ca_kpos_proj(pos)
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)
        tgt2 = self.cross_attn(query=q, key=k, value=v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Transformer(nn.Module):

    def __init__(self, first_layer, n_layers, d_model=512, nhead=8, num_queries=2, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise', num_patterns=0, modulate_t_attn=True, bbox_embed_diff_each_layer=False, args=None):
        super().__init__()
        self.args = args
        mcls_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        mcls_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.mcls_encoder = TransformerEncoder(mcls_encoder_layer, args.moment_layers, mcls_encoder_norm)
        t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, self.args.num_dummies)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.t2v_encoder = TransformerCATEEncoder(t2v_encoder_layer, args.t2v_layers, encoder_norm)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec, d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type, modulate_t_attn=modulate_t_attn, bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        dim = 4096
        llama_default_config = {'dim': dim, 'multiple_of': 256, 'n_heads': 32, 'n_layers': n_layers, 'norm_eps': 1e-06, 'vocab_size': -1, 'first_layer': first_layer}
        self.llama = LLaMATransformer(llama_default_config)
        self.llama_dim_mapper1 = nn.Linear(d_model, dim, bias=False)
        self.llama_dim_mapper2 = nn.Linear(dim, d_model, bias=False)
        ckpt_path = 'D:\\fletcher\\LLMEPET\\llama\\consolidated.00.pth'
        start_time = time.time()
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.llama.custom_load_state_dict(checkpoint, tail=True, strict=False)
        None
        for param in self.llama.parameters():
            param.requires_grad = False

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, video_length=None, moment_idx=None, msrc=None, mpos=None, mmask=None, nmsrc=None, nmpos=None, nmmask=None, ctxtoken=None, gtoken=None, gpos=None, vlen=None):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src
            video length: feature shape
            vlen: actual video length
        Returns:
        """
        if msrc is not None:
            msrc = msrc.permute(1, 0, 2)
            mpos = mpos.permute(1, 0, 2)
            mmemory = self.mcls_encoder(msrc, src_key_padding_mask=mmask, pos=mpos)
            mmemory_moment, mmemory_frames = mmemory[0], mmemory[1:]
        else:
            mmemory_moment = None
            mmemory_frames = None
        if nmsrc is not None:
            nmsrc = nmsrc.permute(1, 0, 2)
            nmpos = nmpos.permute(1, 0, 2)
            nmmemory = self.mcls_encoder(nmsrc, src_key_padding_mask=nmmask, pos=nmpos)
            nmmemory_moment, nmmemory_frames = nmmemory[0], nmmemory[1:]
        else:
            nmmemory_moment = None
            nmmemory_frames = None
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        refpoint_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        t2v_src, attn_weights = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length)
        ctx_src_ = ctxtoken.permute(1, 0, 2)
        fr_token_sim = torch.softmax(torch.matmul(F.normalize((src[:video_length] - ctx_src_).permute(1, 0, 2), dim=2), F.normalize(gtoken, dim=1).T), dim=-1)
        frame_importance = attn_weights[:, :, self.args.num_dummies:].sum(2).clone().detach()
        for i in range(len(frame_importance)):
            frame_importance[i][vlen[i]:] *= 0.0
        frame_importance = frame_importance / frame_importance.sum(1).unsqueeze(1) * frame_importance.size(1)
        fr_token_sim = fr_token_sim * frame_importance.unsqueeze(2).repeat(1, 1, fr_token_sim.size(2))
        fr_token_sim = fr_token_sim.mean(1)
        topk_val, topkidx = torch.topk(fr_token_sim, k=self.args.num_prompts, dim=1)
        src_ = torch.zeros((len(fr_token_sim), self.d_model))
        for i in range(len(fr_token_sim)):
            src_[i] = (topk_val[i].unsqueeze(1) * gtoken[topkidx[i]]).sum(0)
        src_ = src_.reshape(1, src.size(1), -1)
        src_ = src_ + ctx_src_
        pos_ = gpos.reshape([1, 1, self.d_model]).repeat(1, pos_embed.shape[1], 1)
        mask_ = torch.tensor([[False]]).repeat(mask.shape[0], 1)
        src_, _ = self.t2v_encoder(src_, src_key_padding_mask=mask_, pos=pos_, video_length=video_length, dummy=False)
        src = torch.cat([src_, t2v_src], dim=0)
        mask = torch.cat([mask_, mask], dim=1)
        pos_embed = torch.cat([pos_, pos_embed], dim=0)
        src = src[:video_length + 1]
        mask = mask[:, :video_length + 1]
        pos_embed = pos_embed[:video_length + 1]
        memory_trans = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        tmp_src = self.llama_dim_mapper1(memory_trans)
        llama_encoder = self.llama(tmp_src)
        memory_llama = self.llama_dim_mapper2(llama_encoder)
        memory = memory_llama
        memory_global, memory_local = memory[0], memory[1:]
        memory_local += memory_global.unsqueeze(0).repeat(memory_local.size(0), 1, 1)
        mask_local = mask[:, 1:]
        pos_embed_local = pos_embed[1:]
        tgt = torch.zeros(refpoint_embed.shape[0], bs, d)
        hs, references = self.decoder(tgt, memory_local, memory_key_padding_mask=mask_local, pos=pos_embed_local, refpoints_unsigmoid=refpoint_embed)
        memory_local = memory_local.transpose(0, 1)
        return hs, references, memory_local, memory_global, attn_weights, mmemory_moment, nmmemory_moment, mmemory_frames, nmmemory_frames


class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src

    def forward_pre(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayerThin(nn.Module):
    """removed intermediate layer"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(dropout)
        self.dropout2 = DropPath(dropout)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.linear1(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'config': _mock_config(n_heads=4, dim=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'multiple_of': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HungarianEventMatcher,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (LLaMATransformer,
     lambda: ([], {'config': _mock_config(n_layers=1, first_layer=1, dim=4, norm_eps=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LinearLayer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionEmbeddingLearned,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionEmbeddingSine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (RMSNorm,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TrainablePositionalEncoding,
     lambda: ([], {'max_position_embeddings': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerBlock,
     lambda: ([], {'layer_id': 1, 'config': _mock_config(n_heads=4, dim=4, multiple_of=4, norm_eps=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransformerDecoderLayerThin,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransformerEncoderLayerThin,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_fletcherjiang_LLMEPET(_paritybench_base):
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

