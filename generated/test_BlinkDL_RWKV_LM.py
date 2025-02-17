import sys
_module = sys.modules[__name__]
del sys
src = _module
model = _module
trainer = _module
utils = _module
train = _module
run = _module
model = _module
model_run = _module
trainer = _module
utils = _module
train = _module
run = _module
model = _module
model_run = _module
trainer = _module
utils = _module
train = _module
verify = _module
run = _module
binidx = _module
model = _module
model_run = _module
trainer = _module
utils = _module
train = _module
verify = _module
chat = _module
img_demoAE = _module
run = _module
run = _module
binidx = _module
dataset = _module
model = _module
model_img = _module
model_run = _module
trainer = _module
utils = _module
train = _module
verify = _module
make_data = _module
rwkv_v6_demo = _module
binidx = _module
dataset = _module
model = _module
trainer = _module
utils = _module
tokenizer = _module
rwkv_tokenizer = _module
train = _module
rwkv_mmlu_eval = _module
rwkv_v7_demo = _module
rwkv_v7_demo_fast = _module
rwkv_v7_demo_rnn = _module

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


import math


import logging


import torch


import torch.nn as nn


from torch.nn import functional as F


import numpy as np


import torch.optim as optim


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.data.dataloader import DataLoader


import random


import time


from torch.utils.data import Dataset


import types


import copy


from torch.utils.cpp_extension import load


from functools import lru_cache


from itertools import accumulate


import torchvision as vision


import torchvision.transforms as transforms


import torch.nn.functional as F


from typing import List


from typing import Dict


from torch.utils.data import DataLoader


MyFunction = torch.jit.script_method


MyModule = torch.jit.ScriptModule


class WKV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return y
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return y.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        else:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return None, None, None, gw, gu, gk, gv
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return None, None, None, gw.half(), gu.half(), gk.half(), gv.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16()


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)


class RWKV_ChannelMix(MyModule):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - layer_id / args.n_layer
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class RWKV_TinyAttn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_attn = config.rwkv_tiny_attn
        self.n_head = config.rwkv_tiny_head
        self.head_size = self.d_attn // self.n_head
        self.qkv = nn.Linear(config.n_embd, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, config.n_embd)

    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        qk = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_size))
        qk = qk.masked_fill(mask == 0, float('-inf'))
        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v
        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(qkv)


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[..., :q.shape[-2], :], sin[..., :q.shape[-2], :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class MHA_rotary(nn.Module):

    def __init__(self, config, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head
        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.register_buffer('mask', torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)
        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        x = att @ v
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        x = self.output(x)
        return x


class GeGLU(torch.nn.Module):

    def __init__(self, config, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id
        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        hidden_sz = 3 * config.n_ffn
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        k = self.key(x)
        v = self.value(x)
        y = self.weight(F.gelu(k) * v)
        return y


class MHA_pro(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head
        self.time_w = nn.Parameter(torch.ones(self.n_head, config.ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
        self.register_buffer('mask', torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)
        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False)
        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = att * w
        att = self.head_mix(att)
        x = att @ v
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        x = self.output(x) * self.time_gamma[:T, :]
        return x


class RMSNorm(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1.0 / 2)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return self.weight * x_normed


class FixedNorm(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1.0 / 2)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return x_normed


class RWKV_CMix_x070(MyModule):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


DTYPE = torch.half


class WKV_7(torch.autograd.Function):

    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [r, w, k, v, a, b])
            assert all(x.is_contiguous() for x in [r, w, k, v, a, b])
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y


def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)


class RWKV_Tmix_x070(MyModule):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))
        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=0.00064)

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + xv @ self.v1 @ self.v2)
        a = torch.sigmoid(self.a0 + xa @ self.a1 @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        x = RWKV7_OP(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class Block(MyModule):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(args.n_embd)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    @MyFunction
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        factor = 0.0001 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return grad_output, gy


RWKV_HEAD_QK_DIM = 0


def RWKV_Init(model, args):
    None
    None
    for mm in model.modules():
        if 'RecursiveScriptModule' in str(type(mm)):
            if mm.original_name not in ['Linear']:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == 'weight':
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in model.named_parameters():
                if id(ww) == id(parameter):
                    break
            shape = ww.shape
            gain = 1.0
            scale = 1.0
            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:
                    scale = 0.0001
                else:
                    scale = 0
            if isinstance(m, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:
                    scale = 0.5
            if hasattr(m, 'scale_init'):
                scale = m.scale_init
            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)


logger = logging.getLogger(__name__)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer('copy_mask', torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        self.ctx_len = config.ctx_len
        try:
            if os.environ['RWKV_LOAD_MODEL'] == str(False):
                RWKV_Init(self, config)
        except:
            pass
        logger.info('number of parameters: %e', sum(p.numel() for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1e-05)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [{'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0}]
        try:
            optimizer = FusedAdam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        except:
            None
            optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)
        return optimizer

    def forward(self, idx, targets=None):
        idx = idx
        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, 'Cannot forward, because len(input) > model ctx_len.'
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = q @ k.transpose(-2, -1) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size)
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).half()
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).bfloat16()
            x = self.head(x) + c
        else:
            x = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, x)


class RWKV_GPT(nn.Module):

    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, vocab_size, n_layer, n_embd, ctx_len):
        global RWKV_CFG
        super().__init__()
        RWKV_CFG.RUN_DEVICE = RUN_DEVICE
        RWKV_CFG.model_type = model_type
        RWKV_CFG.vocab_size = vocab_size
        RWKV_CFG.n_layer = n_layer
        RWKV_CFG.n_embd = n_embd
        RWKV_CFG.ctx_len = ctx_len
        None
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer('copy_mask', torch.tril(torch.ones(ctx_len, ctx_len)))
        self.ctx_len = ctx_len
        self.eval()
        self.load_state_dict(torch.load(MODEL_NAME + '.pth'))
        self.eval()

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.ctx_len, 'Cannot forward, because len(input) > model ctx_len.'
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = q @ k.transpose(-2, -1) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size)
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size).half()
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size).bfloat16()
            x = self.head(x) + c
        else:
            x = self.head(x)
        return x


class R_ENCODER(MyModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        dd = 8
        self.Bxx = nn.BatchNorm2d(dd * 64)
        self.CIN = nn.Conv2d(3, dd, kernel_size=3, padding=1)
        self.Cx0 = nn.Conv2d(dd, 32, kernel_size=3, padding=1)
        self.Cx1 = nn.Conv2d(32, dd, kernel_size=3, padding=1)
        self.B00 = nn.BatchNorm2d(dd * 4)
        self.C00 = nn.Conv2d(dd * 4, 256, kernel_size=3, padding=1)
        self.C01 = nn.Conv2d(256, dd * 4, kernel_size=3, padding=1)
        self.C02 = nn.Conv2d(dd * 4, 256, kernel_size=3, padding=1)
        self.C03 = nn.Conv2d(256, dd * 4, kernel_size=3, padding=1)
        self.B10 = nn.BatchNorm2d(dd * 16)
        self.C10 = nn.Conv2d(dd * 16, 256, kernel_size=3, padding=1)
        self.C11 = nn.Conv2d(256, dd * 16, kernel_size=3, padding=1)
        self.C12 = nn.Conv2d(dd * 16, 256, kernel_size=3, padding=1)
        self.C13 = nn.Conv2d(256, dd * 16, kernel_size=3, padding=1)
        self.B20 = nn.BatchNorm2d(dd * 64)
        self.C20 = nn.Conv2d(dd * 64, 256, kernel_size=3, padding=1)
        self.C21 = nn.Conv2d(256, dd * 64, kernel_size=3, padding=1)
        self.C22 = nn.Conv2d(dd * 64, 256, kernel_size=3, padding=1)
        self.C23 = nn.Conv2d(256, dd * 64, kernel_size=3, padding=1)
        self.COUT = nn.Conv2d(dd * 64, args.my_img_bit, kernel_size=3, padding=1)

    @MyFunction
    def forward(self, img):
        ACT = F.mish
        x = self.CIN(img)
        xx = self.Bxx(F.pixel_unshuffle(x, 8))
        x = x + self.Cx1(ACT(self.Cx0(x)))
        x = F.pixel_unshuffle(x, 2)
        x = x + self.C01(ACT(self.C00(ACT(self.B00(x)))))
        x = x + self.C03(ACT(self.C02(x)))
        x = F.pixel_unshuffle(x, 2)
        x = x + self.C11(ACT(self.C10(ACT(self.B10(x)))))
        x = x + self.C13(ACT(self.C12(x)))
        x = F.pixel_unshuffle(x, 2)
        x = x + self.C21(ACT(self.C20(ACT(self.B20(x)))))
        x = x + self.C23(ACT(self.C22(x)))
        x = self.COUT(x + xx)
        return torch.sigmoid(x)


class R_DECODER(MyModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        dd = 8
        self.CIN = nn.Conv2d(args.my_img_bit, dd * 64, kernel_size=3, padding=1)
        self.B00 = nn.BatchNorm2d(dd * 64)
        self.C00 = nn.Conv2d(dd * 64, 256, kernel_size=3, padding=1)
        self.C01 = nn.Conv2d(256, dd * 64, kernel_size=3, padding=1)
        self.C02 = nn.Conv2d(dd * 64, 256, kernel_size=3, padding=1)
        self.C03 = nn.Conv2d(256, dd * 64, kernel_size=3, padding=1)
        self.B10 = nn.BatchNorm2d(dd * 16)
        self.C10 = nn.Conv2d(dd * 16, 256, kernel_size=3, padding=1)
        self.C11 = nn.Conv2d(256, dd * 16, kernel_size=3, padding=1)
        self.C12 = nn.Conv2d(dd * 16, 256, kernel_size=3, padding=1)
        self.C13 = nn.Conv2d(256, dd * 16, kernel_size=3, padding=1)
        self.B20 = nn.BatchNorm2d(dd * 4)
        self.C20 = nn.Conv2d(dd * 4, 256, kernel_size=3, padding=1)
        self.C21 = nn.Conv2d(256, dd * 4, kernel_size=3, padding=1)
        self.C22 = nn.Conv2d(dd * 4, 256, kernel_size=3, padding=1)
        self.C23 = nn.Conv2d(256, dd * 4, kernel_size=3, padding=1)
        self.Cx0 = nn.Conv2d(dd, 32, kernel_size=3, padding=1)
        self.Cx1 = nn.Conv2d(32, dd, kernel_size=3, padding=1)
        self.COUT = nn.Conv2d(dd, 3, kernel_size=3, padding=1)

    @MyFunction
    def forward(self, code):
        ACT = F.mish
        x = self.CIN(code)
        x = x + self.C01(ACT(self.C00(ACT(self.B00(x)))))
        x = x + self.C03(ACT(self.C02(x)))
        x = F.pixel_shuffle(x, 2)
        x = x + self.C11(ACT(self.C10(ACT(self.B10(x)))))
        x = x + self.C13(ACT(self.C12(x)))
        x = F.pixel_shuffle(x, 2)
        x = x + self.C21(ACT(self.C20(ACT(self.B20(x)))))
        x = x + self.C23(ACT(self.C22(x)))
        x = F.pixel_shuffle(x, 2)
        x = x + self.Cx1(ACT(self.Cx0(x)))
        x = self.COUT(x)
        return torch.sigmoid(x)


class RWKV_RNN(MyModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cuda')
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        keys = list(z.keys())
        for k in keys:
            if k.endswith('att.w0'):
                z[k] = z[k].float()
            else:
                z[k] = z[k]
            z[k] = z[k].squeeze()
            if k.endswith('att.r_k'):
                z[k] = z[k].flatten()
        assert self.head_size == args.head_size
        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0']
        z['blocks.0.att.v1'] = z['blocks.0.att.a1']
        z['blocks.0.att.v2'] = z['blocks.0.att.a2']

    @MyFunction
    def forward(self, token: 'int', state: 'List[torch.Tensor]'):
        with torch.no_grad():
            z = self.z
            x = z['emb.weight'][token]
            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'
                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + 'ln1.weight'], bias=z[bbb + 'ln1.bias'])
                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = time_mixing(i, self.n_head, self.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1], z[att + 'x_r'], z[att + 'x_w'], z[att + 'x_k'], z[att + 'x_v'], z[att + 'x_a'], z[att + 'x_g'], z[att + 'w0'], z[att + 'w1'], z[att + 'w2'], z[att + 'a0'], z[att + 'a1'], z[att + 'a2'], z[att + 'v0'], z[att + 'v1'], z[att + 'v2'], z[att + 'g1'], z[att + 'g2'], z[att + 'k_k'], z[att + 'k_a'], z[att + 'r_k'], z[att + 'key.weight'], z[att + 'value.weight'], z[att + 'receptance.weight'], z[att + 'output.weight'], z[att + 'ln_x.weight'], z[att + 'ln_x.bias'])
                x = x + xx
                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + 'ln2.weight'], bias=z[bbb + 'ln2.bias'])
                xx, state[i * 3 + 2] = channel_mixing(xx, state[i * 3 + 2], z[ffn + 'x_k'], z[ffn + 'key.weight'], z[ffn + 'value.weight'])
                x = x + xx
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = z['head.weight'] @ x
            return x, state


class L2pooling(nn.Module):

    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class DISTS(torch.nn.Module):

    def __init__(self, load_weights=True):
        super(DISTS, self).__init__()
        vgg_pretrained_features = vision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_buffer('alpha', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_buffer('beta', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)
        weights = torch.load('test/DISTS_weights.pt')
        self.alpha.data = weights['alpha']
        self.beta.data = weights['beta']
        for param in self.parameters():
            param.requires_grad = False

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, require_grad=False, batch_average=False):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-06
        c2 = 1e-06
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)
            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)
        score = 1 - (dist1 + dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score


    class ToBinary(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            return torch.floor(x + 0.5)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.clone()


class WKV_6(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert args.head_size_a == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ctx.save_for_backward(r, k, v, w, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            wkv6_cuda.forward(B, T, C, H, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return None, None, None, None, gr, gk, gv, gw, gu


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


class RWKV_Tmix_x060(MyModule):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - layer_id / args.n_layer
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))
            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))
            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - n / (args.dim_att - 1)) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=1e-05 * args.head_size_divisor ** 2)

    @MyFunction
    def forward(self, x):
        B, T, C = x.size()
        xx = self.time_shift(x) - x
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))
        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww
        x = RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)
        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x


class RWKV_CMix_x060(MyModule):

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - layer_id / args.n_layer
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 3))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 3))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class RWKV(nn.Module):

    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        x = self.head(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FixedNorm,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeGLU,
     lambda: ([], {'config': _mock_config(n_ffn=4, n_embd=4), 'layer_id': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MHA_pro,
     lambda: ([], {'config': _mock_config(n_attn=4, n_head=4, ctx_len=4, n_embd=4), 'layer_id': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MHA_rotary,
     lambda: ([], {'config': _mock_config(n_attn=4, n_head=4, ctx_len=4, n_embd=4), 'layer_id': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RMSNorm,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RWKV_TinyAttn,
     lambda: ([], {'config': _mock_config(rwkv_tiny_attn=4, rwkv_tiny_head=4, n_embd=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RotaryEmbedding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_BlinkDL_RWKV_LM(_paritybench_base):
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

