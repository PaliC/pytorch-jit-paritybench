import sys
_module = sys.modules[__name__]
del sys
datasets = _module
dataset = _module
dataset_bbox = _module
sam_fact_tt_image_encoder = _module
sam_fact_tt_image_encoder_bbox = _module
segment_anything = _module
automatic_mask_generator = _module
build_sam = _module
modeling = _module
common = _module
image_encoder = _module
mask_decoder = _module
mask_decoder_bbox = _module
prompt_encoder = _module
sam = _module
sam_bbox = _module
transformer = _module
predictor = _module
utils = _module
amg = _module
onnx = _module
transforms = _module
test = _module
train = _module
trainer = _module
trainer_bbox = _module
utils = _module
util_script_btcv = _module
util_script_endovis18 = _module
util_script_prostateMRI = _module

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


import random


import numpy as np


import torch


from scipy import ndimage


from scipy.ndimage.interpolation import zoom


from torch.utils.data import Dataset


import pandas as pd


import matplotlib.pyplot as plt


import math


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from torch.nn.parameter import Parameter


from typing import Optional


from typing import Tuple


from typing import Type


from torchvision.ops.boxes import batched_nms


from torchvision.ops.boxes import box_area


from typing import Any


from typing import Dict


from typing import List


from torch.nn import functional as F


from functools import partial


from torch import nn


from copy import deepcopy


from itertools import product


from typing import Generator


from typing import ItemsView


from torchvision.transforms.functional import resize


from torchvision.transforms.functional import to_pil_image


import logging


from torch.utils.data import DataLoader


import torch.backends.cudnn as cudnn


from scipy.ndimage import zoom


import time


import torch.optim as optim


from torch.nn.modules.loss import CrossEntropyLoss


from torchvision import transforms


from torch.optim.lr_scheduler import LambdaLR


class _Fact_tt_ImageEncoderViT(nn.Module):

    def __init__(self, ImageEncoderViT: 'nn.Module', FacTu: 'nn.Module', FacTv: 'nn.Module'):
        super().__init__()
        self.ImageEncoderViT = ImageEncoderViT
        self.FacTu = FacTu
        self.FacTv = FacTv
        self.img_size = self.ImageEncoderViT.img_size

    def forward(self, x: 'torch.Tensor', d_size) ->torch.Tensor:
        x = self.ImageEncoderViT.patch_embed(x)
        if self.ImageEncoderViT.pos_embed is not None:
            x = x + self.ImageEncoderViT.pos_embed
        for blk in self.ImageEncoderViT.blocks:
            x = blk(x, self.FacTu, self.FacTv, d_size)
        x = self.ImageEncoderViT.neck(x.permute(0, 3, 1, 2))
        return x


def window_partition(x: 'torch.Tensor', window_size: 'int') ->Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: 'torch.Tensor', window_size: 'int', pad_hw: 'Tuple[int, int]', hw: 'Tuple[int, int]') ->torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class _Fact_tt_Block(nn.Module):

    def __init__(self, Block: 'nn.Module'):
        super().__init__()
        self.Block = Block

    def forward(self, x: 'torch.Tensor', FacTu, FacTv, d_size) ->torch.Tensor:
        b_size, hw_size = x.shape[0], x.shape[1]
        shortcut = x
        x = self.Block.adapter_norm(x)
        x = self.Block.adapter_linear_down(x)
        x = x.contiguous().view(int(b_size / d_size), d_size, hw_size, hw_size, self.Block.adapter_channels)
        x = torch.permute(x, (0, -1, 1, 2, 3))
        x = self.Block.adapter_conv(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(b_size, hw_size, hw_size, self.Block.adapter_channels)
        x = self.Block.adapter_act(x)
        x = self.Block.adapter_linear_up(x)
        x = shortcut + x
        shortcut = x
        x = self.Block.norm1(x)
        if self.Block.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.Block.window_size)
        x = self.Block.attn(x, FacTu, FacTv)
        if self.Block.window_size > 0:
            x = window_unpartition(x, self.Block.window_size, pad_hw, (H, W))
        x = shortcut + x
        shortcut = x
        x = self.Block.adapter_norm_2(x)
        x = self.Block.adapter_linear_down_2(x)
        x = x.contiguous().view(int(b_size / d_size), d_size, hw_size, hw_size, self.Block.adapter_channels)
        x = torch.permute(x, (0, -1, 1, 2, 3))
        x = self.Block.adapter_conv_2(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(b_size, hw_size, hw_size, self.Block.adapter_channels)
        x = self.Block.adapter_act_2(x)
        x = self.Block.adapter_linear_up_2(x)
        x = shortcut + x
        x = x + self.Block.mlp(self.Block.norm2(x))
        return x


def get_rel_pos(q_size: 'int', k_size: 'int', rel_pos: 'torch.Tensor') ->torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode='linear')
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: 'torch.Tensor', q: 'torch.Tensor', rel_pos_h: 'torch.Tensor', rel_pos_w: 'torch.Tensor', q_size: 'Tuple[int, int]', k_size: 'Tuple[int, int]') ->torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn


class _Fact_tt_Attention(nn.Module):

    def __init__(self, Attention: 'nn.Module'):
        super().__init__()
        self.Attention = Attention

    def forward(self, x: 'torch.Tensor', FacTu, FacTv) ->torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.Attention.qkv(x, FacTu, FacTv).reshape(B, H * W, 3, self.Attention.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.Attention.num_heads, H * W, -1).unbind(0)
        attn = q * self.Attention.scale @ k.transpose(-2, -1)
        if self.Attention.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.Attention.rel_pos_h, self.Attention.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.Attention.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.Attention.proj(x)
        return x


class _Fact_tt_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(self, qkv: 'nn.Module', q_FacTs: 'nn.Module', v_FacTs: 'nn.Module', s):
        super().__init__()
        self.qkv = qkv
        self.q_FacTs = q_FacTs
        self.v_FacTs = v_FacTs
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.dp_q = nn.Dropout(0.1)
        self.dp_v = nn.Dropout(0.1)
        self.s = s

    def forward(self, x, FacTu, FacTv):
        qkv = self.qkv(x)
        new_q = FacTv(self.dp_q(self.q_FacTs(FacTu(x))))
        new_v = FacTv(self.dp_v(self.v_FacTs(FacTu(x))))
        qkv[:, :, :, :self.dim] += new_q * self.s
        qkv[:, :, :, -self.dim:] += new_v * self.s
        return qkv


class Fact_tt_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of FacT_tt
        num_classes: how many classes the model output, default to the vit model
        FacT_tt_layer: which layer we apply FacT_tt.

    """

    def __init__(self, sam_model: 'Sam', r: 'int', fact_layer=None, s=1):
        super(Fact_tt_Sam, self).__init__()
        assert r > 0
        base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        if fact_layer:
            self.fact_layer = fact_layer
        else:
            self.fact_layer = list(range(len(sam_model.image_encoder.blocks)))
        self.q_FacTs = []
        self.v_FacTs = []
        self.FacTu = nn.Linear(base_vit_dim, r, bias=False)
        self.FacTv = nn.Linear(r, base_vit_dim, bias=False)
        nn.init.zeros_(self.FacTv.weight)
        for k, v in sam_model.image_encoder.named_parameters():
            if not '.adapter_' in k:
                v.requires_grad = False
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.fact_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            q_FacTs = nn.Linear(r, r, bias=False)
            v_FacTs = nn.Linear(r, r, bias=False)
            self.q_FacTs.append(q_FacTs)
            self.v_FacTs.append(v_FacTs)
            blk.attn.qkv = _Fact_tt_qkv(w_qkv_linear, q_FacTs, v_FacTs, s)
            blk.attn = _Fact_tt_Attention(blk.attn)
            sam_model.image_encoder.blocks[t_layer_i] = _Fact_tt_Block(blk)
        sam_model.image_encoder = _Fact_tt_ImageEncoderViT(sam_model.image_encoder, self.FacTu, self.FacTv)
        self.sam = sam_model

    def save_parameters(self, filename: 'str') ->None:
        """Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both FacT_tt and fc parameters.
        """
        assert filename.endswith('.pt') or filename.endswith('.pth')
        num_layer = len(self.q_FacTs)
        a_tensors = {f'q_FacTs_{i:03d}': self.q_FacTs[i].weight for i in range(num_layer)}
        b_tensors = {f'v_FacTs_{i:03d}': self.v_FacTs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        adapter_tensor = {}
        FacTu_tensors = {}
        FacTv_tensors = {}
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if '.adapter_' in key:
                adapter_tensor[key] = value
            if 'FacTu' in key:
                FacTu_tensors[key] = value
            if 'FacTv' in key:
                FacTv_tensors[key] = value
        merged_dict = {**a_tensors, **b_tensors, **FacTu_tensors, **FacTv_tensors, **prompt_encoder_tensors, **mask_decoder_tensors, **adapter_tensor}
        torch.save(merged_dict, filename)

    def load_parameters(self, filename: 'str') ->None:
        """Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\\

        load both FacT_tt and fc parameters.
        """
        assert filename.endswith('.pt') or filename.endswith('.pth')
        state_dict = torch.load(filename)
        for i, q_FacTs in enumerate(self.q_FacTs):
            saved_key = f'q_FacTs_{i:03d}'
            saved_tensor = state_dict[saved_key]
            q_FacTs.weight = Parameter(saved_tensor)
        for i, v_FacTs in enumerate(self.v_FacTs):
            saved_key = f'v_FacTs_{i:03d}'
            saved_tensor = state_dict[saved_key]
            v_FacTs.weight = Parameter(saved_tensor)
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()
        FacTu_keys = [k for k in sam_keys if 'FacTu' in k]
        FacTu_values = [state_dict[k] for k in FacTu_keys]
        FacTu_new_state_dict = {k: v for k, v in zip(FacTu_keys, FacTu_values)}
        sam_dict.update(FacTu_new_state_dict)
        FacTv_keys = [k for k in sam_keys if 'FacTv' in k]
        FacTv_values = [state_dict[k] for k in FacTv_keys]
        FacTv_new_state_dict = {k: v for k, v in zip(FacTv_keys, FacTv_values)}
        sam_dict.update(FacTv_new_state_dict)
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        adapter_keys = [k for k in sam_keys if '.adapter_' in k]
        adapter_values = [state_dict[k] for k in adapter_keys]
        adapter_new_state_dict = {k: v for k, v in zip(adapter_keys, adapter_values)}
        sam_dict.update(adapter_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) ->None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size, bbox_input):
        return self.sam(batched_input, multimask_output, image_size, bbox_input)


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim: 'int', mlp_dim: 'int', act: 'Type[nn.Module]'=nn.GELU) ->None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: 'int', eps: 'float'=1e-06) ->None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self, embedding_dim: 'int', num_heads: 'int', downsample_rate: 'int'=1) ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, 'num_heads must divide embedding_dim.'
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: 'Tensor', num_heads: 'int') ->Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: 'Tensor') ->Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor') ->Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(self, dim: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, qkv_bias: 'bool'=True, norm_layer: 'Type[nn.Module]'=nn.LayerNorm, act_layer: 'Type[nn.Module]'=nn.GELU, use_rel_pos: 'bool'=False, rel_pos_zero_init: 'bool'=True, window_size: 'int'=0, input_size: 'Optional[Tuple[int, int]]'=None) ->None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size
        self.adapter_channels = 384
        self.adapter_linear_down = nn.Linear(dim, self.adapter_channels, bias=False)
        self.adapter_linear_up = nn.Linear(self.adapter_channels, dim, bias=False)
        self.adapter_conv = nn.Conv3d(self.adapter_channels, self.adapter_channels, kernel_size=(3, 1, 1), padding='same')
        self.adapter_act = nn.GELU()
        self.adapter_norm = norm_layer(dim)
        self.adapter_linear_down_2 = nn.Linear(dim, self.adapter_channels, bias=False)
        self.adapter_linear_up_2 = nn.Linear(self.adapter_channels, dim, bias=False)
        self.adapter_conv_2 = nn.Conv3d(self.adapter_channels, self.adapter_channels, kernel_size=(3, 1, 1), padding='same')
        self.adapter_act_2 = nn.GELU()
        self.adapter_norm_2 = norm_layer(dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, kernel_size: 'Tuple[int, int]'=(16, 16), stride: 'Tuple[int, int]'=(16, 16), padding: 'Tuple[int, int]'=(0, 0), in_chans: 'int'=3, embed_dim: 'int'=768) ->None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class ImageEncoderViT(nn.Module):

    def __init__(self, img_size: 'int'=1024, patch_size: 'int'=16, in_chans: 'int'=3, embed_dim: 'int'=768, depth: 'int'=12, num_heads: 'int'=12, mlp_ratio: 'float'=4.0, out_chans: 'int'=256, qkv_bias: 'bool'=True, norm_layer: 'Type[nn.Module]'=nn.LayerNorm, act_layer: 'Type[nn.Module]'=nn.GELU, use_abs_pos: 'bool'=True, use_rel_pos: 'bool'=False, rel_pos_zero_init: 'bool'=True, window_size: 'int'=0, global_attn_indexes: 'Tuple[int, ...]'=()) ->None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed: 'Optional[nn.Parameter]' = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, window_size=window_size if i not in global_attn_indexes else 0, input_size=(img_size // patch_size, img_size // patch_size))
            self.blocks.append(block)
        self.neck = nn.Sequential(nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False), LayerNorm2d(out_chans), nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False), LayerNorm2d(out_chans))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x


class MLP(nn.Module):

    def __init__(self, input_dim: 'int', hidden_dim: 'int', output_dim: 'int', num_layers: 'int', sigmoid_output: 'bool'=False) ->None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder(nn.Module):

    def __init__(self, *, transformer_dim: int, transformer: nn.Module, num_multimask_outputs: int=3, activation: Type[nn.Module]=nn.GELU, iou_head_depth: int=3, iou_head_hidden_dim: int=256) ->None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 4), activation(), nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 8), activation(), nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 16), activation(), nn.ConvTranspose2d(transformer_dim // 16, transformer_dim // 32, kernel_size=2, stride=2), activation())
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3) for i in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(self, image_embeddings: 'torch.Tensor', image_pe: 'torch.Tensor', sparse_prompt_embeddings: 'torch.Tensor', dense_prompt_embeddings: 'torch.Tensor', multimask_output: 'bool') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(image_embeddings=image_embeddings, image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings)
        return masks, iou_pred

    def predict_masks(self, image_embeddings: 'torch.Tensor', image_pe: 'torch.Tensor', sparse_prompt_embeddings: 'torch.Tensor', dense_prompt_embeddings: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = torch.repeat_interleave(image_embeddings, 1, dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, 1, dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: 'List[torch.Tensor]' = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: 'int'=64, scale: 'Optional[float]'=None) ->None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer('positional_encoding_gaussian_matrix', scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: 'torch.Tensor') ->torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: 'Tuple[int, int]') ->torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: 'Any' = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: 'torch.Tensor', image_size: 'Tuple[int, int]') ->torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)


class PromptEncoder(nn.Module):

    def __init__(self, embed_dim: 'int', image_embedding_size: 'Tuple[int, int]', input_image_size: 'Tuple[int, int]', mask_in_chans: 'int', activation: 'Type[nn.Module]'=nn.GELU) ->None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: 'int' = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = 4 * image_embedding_size[0], 4 * image_embedding_size[1]
        self.mask_downscaling = nn.Sequential(nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans // 4), activation(), nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans), activation(), nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1))
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) ->torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: 'torch.Tensor', labels: 'torch.Tensor', pad: 'bool') ->torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: 'torch.Tensor') ->torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: 'torch.Tensor') ->torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self, points: 'Optional[Tuple[torch.Tensor, torch.Tensor]]', boxes: 'Optional[torch.Tensor]', masks: 'Optional[torch.Tensor]') ->int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) ->torch.device:
        return self.point_embeddings[0].weight.device

    def forward(self, points: 'Optional[Tuple[torch.Tensor, torch.Tensor]]', boxes: 'Optional[torch.Tensor]', masks: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=boxes is None)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings


class Sam(nn.Module):
    mask_threshold: 'float' = 0.0
    image_format: 'str' = 'RGB'

    def __init__(self, image_encoder: 'ImageEncoderViT', prompt_encoder: 'PromptEncoder', mask_decoder: 'MaskDecoder', pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375]) ->None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) ->Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size, bbox_input):
        outputs = self.forward_train(batched_input, multimask_output, image_size, bbox_input)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size, bbox_input):
        b_size, hw_size, d_size = batched_input.shape[0], batched_input.shape[-2], batched_input.shape[1]
        batched_input = batched_input.contiguous().view(-1, 3, hw_size, hw_size)
        bbox_input = bbox_input.contiguous().view(-1, 4)
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images, d_size)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=bbox_input, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=image_embeddings, image_pe=self.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output)
        masks = self.postprocess_masks(low_res_masks, input_size=(image_size, image_size), original_size=(image_size, image_size))
        outputs = {'masks': masks, 'iou_predictions': iou_predictions, 'low_res_logits': low_res_masks}
        return outputs

    def postprocess_masks(self, masks: 'torch.Tensor', input_size: 'Tuple[int, ...]', original_size: 'Tuple[int, ...]') ->torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(masks, (self.image_encoder.img_size, self.image_encoder.img_size), mode='bilinear', align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode='bilinear', align_corners=False)
        return masks

    def preprocess(self, x: 'torch.Tensor') ->torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class TwoWayAttentionBlock(nn.Module):

    def __init__(self, embedding_dim: 'int', num_heads: 'int', mlp_dim: 'int'=2048, activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2, skip_first_layer_pe: 'bool'=False) ->None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: 'Tensor', keys: 'Tensor', query_pe: 'Tensor', key_pe: 'Tensor') ->Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):

    def __init__(self, depth: 'int', embedding_dim: 'int', num_heads: 'int', mlp_dim: 'int', activation: 'Type[nn.Module]'=nn.ReLU, attention_downsample_rate: 'int'=2) ->None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, activation=activation, attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=i == 0))
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: 'Tensor', image_pe: 'Tensor', point_embedding: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


def calculate_stability_score(masks: 'torch.Tensor', mask_threshold: 'float', threshold_offset: 'float') ->torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    intersections = (masks > mask_threshold + threshold_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > mask_threshold - threshold_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


class SamOnnxModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the ONNX export script for details.
    """

    def __init__(self, model: 'Sam', return_single_mask: 'bool', use_stability_score: 'bool'=False, return_extra_metrics: 'bool'=False) ->None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

    @staticmethod
    def resize_longest_image_size(input_image_size: 'torch.Tensor', longest_side: 'int') ->torch.Tensor:
        input_image_size = input_image_size
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5)
        return transformed_size

    def _embed_points(self, point_coords: 'torch.Tensor', point_labels: 'torch.Tensor') ->torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)
        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)
        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[i].weight * (point_labels == i)
        return point_embedding

    def _embed_masks(self, input_mask: 'torch.Tensor', has_mask_input: 'torch.Tensor') ->torch.Tensor:
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (1 - has_mask_input) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

    def mask_postprocessing(self, masks: 'torch.Tensor', orig_im_size: 'torch.Tensor') ->torch.Tensor:
        masks = F.interpolate(masks, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size)
        masks = masks[..., :int(prepadded_size[0]), :int(prepadded_size[1])]
        orig_im_size = orig_im_size
        h, w = orig_im_size[0], orig_im_size[1]
        masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)
        return masks

    def select_masks(self, masks: 'torch.Tensor', iou_preds: 'torch.Tensor', num_points: 'int') ->Tuple[torch.Tensor, torch.Tensor]:
        score_reweight = torch.tensor([[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)])
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)
        return masks, iou_preds

    @torch.no_grad()
    def forward(self, image_embeddings: 'torch.Tensor', point_coords: 'torch.Tensor', point_labels: 'torch.Tensor', mask_input: 'torch.Tensor', has_mask_input: 'torch.Tensor', orig_im_size: 'torch.Tensor'):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)
        masks, scores = self.model.mask_decoder.predict_masks(image_embeddings=image_embeddings, image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embedding, dense_prompt_embeddings=dense_embedding)
        if self.use_stability_score:
            scores = calculate_stability_score(masks, self.model.mask_threshold, self.stability_score_offset)
        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])
        upscaled_masks = self.mask_postprocessing(masks, orig_im_size)
        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(upscaled_masks, self.model.mask_threshold, self.stability_score_offset)
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks
        return upscaled_masks, scores, masks


class DiceLoss(nn.Module):

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = 1.0 * (input_tensor == i)
            temp_prob[input_tensor == -100] = -100
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-05
        mask = target != -100
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'embedding_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (LayerNorm2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPBlock,
     lambda: ([], {'embedding_dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_cchen_cc_MA_SAM(_paritybench_base):
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

