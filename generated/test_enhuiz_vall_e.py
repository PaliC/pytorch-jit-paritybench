import sys
_module = sys.modules[__name__]
del sys
plot = _module
setup = _module
vall_e = _module
config = _module
data = _module
emb = _module
g2p = _module
qnt = _module
export = _module
sampler = _module
train = _module
ar = _module
base = _module
nar = _module

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


import copy


import logging


import random


from collections import defaultdict


from functools import cache


from functools import cached_property


from itertools import groupby


from itertools import zip_longest


from typing import Any


import numpy as np


from torch import Tensor


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import string


import torchaudio


import math


from functools import partial


from typing import Literal


from typing import overload


import torch.nn.functional as F


from torch import einsum


from torch import nn


from torch.distributions import Categorical


from torch.nn.utils.rnn import pad_sequence


from torch.utils.checkpoint import checkpoint


class SinusodialEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        omega = torch.exp(-math.log(10000.0) * exponent)
        self.omega: 'torch.Tensor'
        self.register_buffer('omega', omega, persistent=False)

    @property
    def d_half(self):
        assert self.d_model % 2 == 0, 'Only support even d_model.'
        return self.d_model // 2

    def forward(self, x):
        """
        Args:
            x: (...)
        Returns:
            pe: (... d)
        """
        omega = self.omega
        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)
        x = x.unsqueeze(-1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return x

    def get_pe(self, n: 'int'):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        Args:
            x: (b t c)
        """
        e = self.get_pe(x.shape[1])
        e = e[None]
        x = x + e
        return x


class Attention(nn.Module):

    def __init__(self, d_model, n_heads, casual):
        super().__init__()
        assert d_model % n_heads == 0
        dim_head = d_model // n_heads
        self.casual = casual
        self.n_heads = n_heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, m):
        """
        Args:
            x: (b t c)
            m: (b t c), 1 is data, 0 is padding
        Returns:
            x: (b t c)
        """
        h = self.n_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b t h d', h=h), (q, k, v))
        e = einsum('b i h d, b j h d -> b i j h', q, k)
        e = e * self.scale
        kpm = m.unsqueeze(1) * m.unsqueeze(2)
        if self.casual:
            kpm = kpm.squeeze(-1).tril().unsqueeze(-1)
        e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)
        a = e.softmax(dim=2)
        o = einsum('b i j h, b j h d -> b i h d', a, v)
        o = o.flatten(-2)
        o = self.to_out(o)
        o = o * m
        return o


class AdaLN(nn.Module):

    def __init__(self, d_model, n_levels, eps=1e-05, k=0.1, c=2):
        super().__init__()
        self.eps = eps
        self.emb = nn.Embedding(n_levels, d_model * 2)
        self.k = k
        self.c = c
        nn.init.zeros_(self.emb.weight)

    def forward(self, x, l):
        logγ, β = self.emb(l).unsqueeze(1).chunk(2, dim=-1)
        h = F.layer_norm(x, x.shape[-1:], eps=self.eps)
        h = self.c * (1 - (self.k * h).detach()) * h
        y = logγ.exp() * h + β
        return y


class PrenormResidual(nn.Module):

    def __init__(self, block, d_model, p_dropout, requires_mask=False, norm_type='ln', n_levels: 'int | None'=None):
        super().__init__()
        self.block = block
        self.requires_mask = requires_mask
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == 'adaln':
            assert n_levels is not None
            self.norm = AdaLN(d_model, n_levels)
        else:
            raise NotImplementedError(norm_type)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, m, l):
        """
        Args:
            x: input (b t d)
            m: mask (b t 1), 1 is valuable and 0 is padding
            l: level to use, required only for AdaLN
        """
        nopts = {'l': l} if self.norm_type == 'adaln' else {}
        bopts = {'m': m} if self.requires_mask else {}
        x = x + self.dropout(self.block(self.norm(x, **nopts) * m, **bopts))
        return x * m


class Block(nn.Sequential):

    def __init__(self, d_model, n_heads, p_dropout, casual, norm_type, n_levels):
        super().__init__()
        self.attn = PrenormResidual(Attention(d_model, n_heads, casual), d_model=d_model, p_dropout=p_dropout, requires_mask=True, norm_type=norm_type, n_levels=n_levels)
        self.ffn = PrenormResidual(nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(p_dropout), nn.Linear(d_model * 4, d_model)), d_model=d_model, p_dropout=p_dropout, norm_type=norm_type, n_levels=n_levels)

    def forward(self, x, m, l):
        """
        Args:
            x: (b t c)
            m: (b t 1)
            l: (b)
        """
        poor_in_vram = True
        if x.requires_grad and poor_in_vram:
            x = checkpoint(self.attn, x, m, l)
        else:
            x = self.attn(x, m, l)
        x = self.ffn(x, m, l)
        return x


class Embedding(nn.Embedding):

    def forward(self, x_list: 'list[Tensor]') ->list[Tensor]:
        if len(x_list) == 0:
            return []
        return super().forward(torch.cat(x_list)).split([*map(len, x_list)])


class MultiEmbedding(nn.Module):
    """
    This embedding sums embeddings on different levels.
    """

    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        self.max_n_levels = max_n_levels
        self.n_tokens = n_tokens
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

    def forward(self, x_list: 'list[Tensor]') ->list[Tensor]:
        if len(x_list) == 0:
            return []
        w = self.weight
        padded_x_list = []
        for xi in x_list:
            xi = F.one_hot(xi, num_classes=self.n_tokens)
            xi = F.pad(xi, (0, 0, 0, w.shape[0] - xi.shape[1]))
            padded_x_list.append(xi)
        x = torch.cat(padded_x_list)
        x = einsum('l k d, n l k -> n d', w, x)
        x_list = x.split([*map(len, x_list)])
        return x_list


def _join(x: 'tuple[Tensor]', sep: 'Tensor'):
    """
    Args:
        x: (k t d)
        sep: (d)
    """
    ret = x[0]
    for i in range(1, len(x)):
        ret = torch.cat((ret, sep[None], x[i]), dim=0)
    return ret


def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)
    stop = torch.tensor(l, device=device).unsqueeze(1)
    return (seq < stop).float()


def list_to_tensor(x_list: 'list[Tensor]', pattern='t b c -> b t c'):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    l = list(map(len, x_list))
    x = rearrange(pad_sequence(x_list), pattern)
    m = _create_mask(l, x_list[0].device)
    m = m.t().unsqueeze(-1)
    m = rearrange(m, pattern)
    m = m
    return x, m


class Base(nn.Module):

    @property
    def casual(self) ->bool:
        raise NotImplementedError

    @property
    def n_resp_levels(self) ->int:
        raise NotImplementedError

    @property
    def use_stop_token(self) ->bool:
        raise NotImplementedError

    @property
    def norm_type(self):
        raise NotImplementedError

    @property
    def n_prom_levels(self) ->int:
        return 8

    @property
    def resp_loss_only(self):
        raise NotImplementedError

    def __init__(self, n_tokens: 'int', d_model: 'int'=512, n_heads: 'int'=8, n_layers: 'int'=12, p_dropout: 'float'=0.1):
        super().__init__()
        self.n_tokens = n_tokens
        casual = self.casual
        n_stop_tokens = 1 if self.use_stop_token else 0
        n_resp_tokens = n_tokens + n_stop_tokens
        self.text_emb = Embedding(n_tokens, d_model)
        self.proms_emb = MultiEmbedding(self.n_prom_levels, n_tokens, d_model)
        self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model)
        self.sin_emb = SinusodialEmbedding(d_model)
        self.sep = nn.Parameter(torch.randn(d_model))
        blocks = [Block(d_model=d_model, n_heads=n_heads, p_dropout=p_dropout, casual=casual, norm_type=self.norm_type, n_levels=self.n_resp_levels) for _ in range(n_layers)]
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(d_model, n_resp_tokens)

    @property
    def stop_token(self):
        if not self.use_stop_token:
            raise ValueError('Not using stop token!')
        return self.n_tokens

    @property
    def ignore_index(self):
        return -100

    @staticmethod
    def _samplewise_merge_tensors(*l, sep: (Tensor | None)):
        if sep is None:
            cat = torch.cat
        else:
            cat = partial(_join, sep=sep)
        return [*map(cat, zip(*l))]

    @overload
    def forward(self, text_list: 'list[Tensor]', proms_list: 'list[Tensor]', resps_list: 'list[Tensor]', targ_list: 'list[Tensor] | None'=None, quant_levels: 'Tensor | None'=None, shift_targ_list: 'bool'=False, return_all_resp: 'Literal[False]'=False, sampling_temperature: 'float'=1.0) ->Tensor:
        ...

    @overload
    def forward(self, text_list: 'list[Tensor]', proms_list: 'list[Tensor]', resps_list: 'list[Tensor]', targ_list: 'list[Tensor] | None'=None, quant_levels: 'Tensor | None'=None, shift_targ_list: 'bool'=False, return_all_resp: 'Literal[True]'=True, sampling_temperature: 'float'=1.0) ->list[Tensor]:
        ...

    def forward(self, text_list: 'list[Tensor]', proms_list: 'list[Tensor]', resps_list: 'list[Tensor]', targ_list: 'list[Tensor] | None'=None, quant_levels: 'Tensor | None'=None, shift_targ_list: 'bool'=False, return_all_resp: 'bool'=False, sampling_temperature: 'float'=1.0):
        """
        Args:
            text_list: [t] * b
            proms_list: [t' l] * b, l quantization levels.
            resps_list: [t'' l] * b, l quantization levels.
            targ_list: [t''] * b, one quantization level only, when given, loss will be computed
            quant_levels: specify which quant_levels to feed forward, used in NAR mode.
            shift_targ_list: whether to shift target list when computing loss. True if AR.
            return_all_resp: True if NAR.
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
        Returns:
            y: sampled tokens
        """
        x_list = self._samplewise_merge_tensors(self.text_emb(text_list), self.proms_emb(proms_list), self.resps_emb(resps_list), sep=self.sep)
        x, m = list_to_tensor(x_list)
        x = self.sin_emb.add_pe(x)
        for block in self.blocks:
            x = block(x, m, quant_levels)
        h = self.classifier(x) * m
        h_list = [hi[:li] for hi, li in zip(h, map(len, x_list))]
        if targ_list is not None:
            if any([(l == 0) for l in map(len, targ_list)]):
                raise ValueError('Cannot compute loss given empty targ_list.')
            device = h.device
            ignore_sep = torch.tensor(self.ignore_index, device=device)
            prom_list = [torch.full_like(t[..., 0], self.ignore_index) for t in proms_list]
            text_prom_list = self._samplewise_merge_tensors(text_list, prom_list, sep=ignore_sep)
            for i in range(len(text_prom_list)):
                if self.resp_loss_only:
                    text_prom_list[i][:] = self.ignore_index
                else:
                    text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
                    text_prom_list[i][-1] = self.ignore_index
            if shift_targ_list:
                targ_list = [*targ_list]
                for i in range(len(targ_list)):
                    targ_list[i] = targ_list[i].roll(-1, dims=0)
                    targ_list[i][-1] = self.stop_token
            y_list = self._samplewise_merge_tensors(text_prom_list, targ_list, sep=ignore_sep)
            self.loss = dict(nll=F.cross_entropy(torch.cat(h_list), torch.cat(y_list), ignore_index=self.ignore_index))
        if return_all_resp:
            logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
            ret = [Categorical(logits=hi / sampling_temperature).sample() for hi in logits]
        else:
            logits = torch.stack([hi[-1] for hi in h_list])
            ret = Categorical(logits=logits / sampling_temperature).sample()
        return ret


class NAR(Base):

    @property
    def n_resp_levels(self):
        return 7

    @property
    def casual(self):
        return False

    @property
    def use_stop_token(self):
        return False

    @property
    def norm_type(self):
        return 'adaln'

    @property
    def resp_loss_only(self):
        return True

    def forward(self, text_list: 'list[Tensor]', proms_list: 'list[Tensor]', resps_list: 'list[Tensor]', sampling_temperature: 'float'=0.2):
        """
        Args:
            text_list: [t] * b
            proms_list: [t' l] * b, l=8
            resps_list: [t'' l] * b, l=1 or 8, 1 for testing and 8 for training.
        Returns:
            [t'' l], l=8 if testing. empty list will be returned during training.
        """
        n_levels_set = {r.shape[-1] for r in resps_list}
        if len(n_levels_set) > 1:
            raise ValueError(f'Please give only one level, got {n_levels_set}.')
        n_levels = next(iter(n_levels_set))
        device = text_list[0].device
        if n_levels == self.n_resp_levels + 1:
            assert resps_list is not None
            quant_levels = torch.randint(0, self.n_resp_levels, (len(resps_list),))
            prev_list = [o[..., :l + 1] for o, l in zip(resps_list, quant_levels)]
            targ_list = [o[..., l + 1] for o, l in zip(resps_list, quant_levels)]
            quant_levels = quant_levels
            _ = super().forward(text_list, proms_list, prev_list, targ_list, return_all_resp=True, shift_targ_list=False, quant_levels=quant_levels)
            prev_list = []
        else:
            prev_list = resps_list
            while True:
                level = prev_list[0].shape[-1] - 1
                if level >= self.n_resp_levels:
                    break
                quant_levels = torch.full((len(text_list),), level, device=device)
                resp_list = super().forward(text_list, proms_list, prev_list, return_all_resp=True, shift_targ_list=False, quant_levels=quant_levels, sampling_temperature=sampling_temperature)
                prev_list = [torch.cat([rs, r.unsqueeze(-1)], dim=-1) for rs, r in zip(prev_list, resp_list)]
        return prev_list


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PrenormResidual,
     lambda: ([], {'block': _mock_layer(), 'd_model': 4, 'p_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SinusodialEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_enhuiz_vall_e(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

