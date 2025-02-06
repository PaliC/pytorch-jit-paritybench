import sys
_module = sys.modules[__name__]
del sys
v1 = _module
constants = _module
jax = _module
layers = _module
models = _module
logger = _module
mixins = _module
mlx = _module
layers = _module
models = _module
data = _module
layers = _module
models = _module
utils = _module
hubconf = _module
main_attnprobe = _module
conftest = _module
test_backend = _module
v2 = _module
layers = _module
models = _module
layers = _module
models = _module
layers = _module
models = _module
utils = _module
test_backend = _module

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


import functools


from typing import Callable


from typing import List


from typing import Literal


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


from typing import Any


from typing import Dict


import collections


import logging


import time


from typing import Collection


from typing import Generator


import torch


from torch import distributed as dist


import math


from typing import TYPE_CHECKING


from typing import NoReturn


from torch.utils.data import DataLoader


from torch.utils.data import distributed


from torch import nn


from torch.nn import functional as F


import torch.distributed as dist


from torch.backends import cudnn


import numpy as np


from torch import nn as torch_nn


class PatchEmbed(nn.Module):

    def __init__(self, img_size: 'Union[int, Tuple[int, int]]'=224, patch_size: 'Union[int, Tuple[int, int]]'=16, in_chans: 'int'=3, embed_dim: 'int'=768, norm_layer: 'Optional[Callable[[int], nn.Module]]'=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.img_size, self.embed_dim = img_size, embed_dim
        self.patch_size = patch_size
        self.grid_size = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class SinCosPosEmbed(nn.Module):

    def __init__(self, cls_token: 'bool'=False):
        super().__init__()
        self.cls_token = cls_token

    def forward(self, h: 'int', w: 'int', embed_dim: 'int') ->torch.Tensor:
        assert embed_dim % 2 == 0, embed_dim
        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = torch.concatenate([emb_h, emb_w], dim=1)
        if self.cls_token:
            pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: 'int', pos: 'torch.Tensor') ->torch.Tensor:
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000 ** omega
        pos = pos.reshape(-1)
        out = pos[:, None] * omega[None, :]
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.concatenate([emb_sin, emb_cos], dim=1)
        return emb


class ViTPreprocessor(nn.Module):

    def __init__(self, patchifier: 'PatchEmbed', drop_patches: 'bool'=False, cls_token: 'bool'=False, pos_embed_type: "Literal['sincos', 'absolute']"='sincos'):
        super().__init__()
        self.patchifier = patchifier
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patchifier.embed_dim)) if cls_token else None
        if pos_embed_type == 'sincos':
            self.pos_embed = SinCosPosEmbed(cls_token)
        else:
            shape = 1, self.patchifier.num_patches + cls_token, self.patchifier.embed_dim
            self.pos_embed = nn.Parameter(torch.zeros(shape))
        self.drop_patches = drop_patches
        self.initialize_weights()

    def initialize_weights(self) ->None:
        if not isinstance(self.pos_embed, SinCosPosEmbed):
            torch.nn.init.normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        if hasattr(self.patchifier, 'proj'):
            w = self.patchifier.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        B, _, H, W = x.shape
        tokens = self.patchifier(x)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
        B, N, D = tokens.shape
        if callable(self.pos_embed):
            p_h, p_w = self.patchifier.patch_size
            pos_embed = self.pos_embed(H // p_h, W // p_w, D).unsqueeze(0)
        else:
            pos_embed = self.pos_embed
        pos_embed = pos_embed
        tokens = tokens + pos_embed[:, :N]
        if self.drop_patches and mask is not None:
            if self.cls_token is not None:
                cls_token, tokens = tokens[:, :1], tokens[:, 1:]
            tokens = tokens[~mask].reshape(B, -1, D)
            if self.cls_token is not None:
                tokens = torch.cat([cls_token, tokens], dim=1)
        return tokens


class Attention(nn.Module):

    def __init__(self, dim: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=False, attn_drop: 'float'=0.0, proj_drop: 'float'=0.0, use_bias: 'bool'=True, is_causal: 'bool'=False):
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, **_: Any) ->torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):

    def __init__(self, in_features: 'int', hidden_features: 'Optional[int]'=None, out_features: 'Optional[int]'=None, act_layer: 'Callable[[], nn.Module]'=nn.GELU, use_bias: 'bool'=True, drop: 'float'=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=use_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


LayerNorm = functools.partial(nn.LayerNorm, eps=1e-06)


class Block(nn.Module):

    def __init__(self, dim: 'int', attn_target: 'Callable[[bool], nn.Module]', ffn_target: 'Callable[..., nn.Module]'=MLP, mlp_hidden_dim: 'Optional[int]'=None, act_layer: 'Callable[[], nn.Module]'=nn.GELU, norm_layer: 'Callable[[int], nn.Module]'=LayerNorm, ffn_dropout_rate: 'float'=0.0, use_bias: 'bool'=True):
        assert not isinstance(attn_target, nn.Module), 'attn_target should be a callable. Otherwise attn_target is shared across blocks!'
        assert not isinstance(ffn_target, nn.Module), 'ffn_target should be a callable. Otherwise ffn_target is shared across blocks!'
        super().__init__()
        self.attn = attn_target(use_bias)
        self.norm_1 = norm_layer(dim)
        self.mlp = ffn_target(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=ffn_dropout_rate, use_bias=use_bias)
        self.norm_2 = norm_layer(dim)

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        x = x + self.attn(self.norm_1(x), mask=mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class AverageLayers(nn.Module):

    def __init__(self, layers: 'Sequence[int]', reduce: 'bool'=False):
        super().__init__()
        self.layers = layers
        self.reduce = reduce

    def forward(self, _: 'torch.Tensor', layer_features: 'List[torch.Tensor]') ->torch.Tensor:
        layer_features = [layer_features[layer_id] for layer_id in self.layers]
        feats = torch.stack(layer_features, dim=-1).mean(dim=-1)
        return feats.mean(dim=1) if self.reduce else feats

    @property
    def max_block_id(self) ->int:
        return max(self.layers)


class AttentionPoolingClassifier(nn.Module):

    def __init__(self, dim: 'int', out_features: 'int', num_heads: 'int'=12, num_queries: 'int'=1, use_batch_norm: 'bool'=True, qkv_bias: 'bool'=False, linear_bias: 'bool'=False, average_pool: 'bool'=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, out_features, bias=linear_bias)
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-06) if use_batch_norm else nn.Identity()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B, N, C = x.shape
        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)
        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls
        out = self.linear(x_cls)
        return out


class Transformer(nn.Module):

    def __init__(self, attn_target: 'Callable[[bool], nn.Module]', embed_dim: 'int', num_blocks: 'int', ffn_target: 'Callable[..., nn.Module]'=layers.MLP, post_transformer_layer: 'Optional[nn.Module]'=None, norm_layer: 'Callable[[int], nn.Module]'=layers.LayerNorm, mlp_ratio: 'int'=4, mlp_hidden_dim: 'Optional[int]'=None, ffn_dropout_rate: 'float'=0.0, use_bias: 'bool'=False, post_trunk_norm: 'bool'=True):
        super().__init__()
        if mlp_hidden_dim is None:
            mlp_hidden_dim = int(mlp_ratio * embed_dim)
        self.blocks = nn.ModuleList([layers.Block(dim=embed_dim, attn_target=attn_target, ffn_target=ffn_target, mlp_hidden_dim=mlp_hidden_dim, norm_layer=norm_layer, ffn_dropout_rate=ffn_dropout_rate, use_bias=use_bias) for _ in range(num_blocks)])
        self.post_trunk_norm = norm_layer(embed_dim) if post_trunk_norm else None
        self.post_transformer_layer = post_transformer_layer

    def forward(self, tokens: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, max_block_id: 'Optional[int]'=-1, return_features: 'bool'=False) ->Union[Tuple[torch.Tensor, List[torch.Tensor]], List[torch.Tensor]]:
        if max_block_id is None:
            assert self.post_transformer_layer is not None, 'Unable to determine the max block id.'
            max_block_id = self.post_transformer_layer.max_block_id
        features = []
        for blk_id, blk in enumerate(self.blocks):
            tokens = blk(tokens, mask=mask)
            features.append(tokens)
            if blk_id == max_block_id:
                break
        if return_features:
            return features
        if self.post_trunk_norm is not None:
            tokens = self.post_trunk_norm(tokens)
        if self.post_transformer_layer is not None:
            tokens = self.post_transformer_layer(tokens, layer_features=features)
        return tokens, features


class PrefixCausalAttention(Attention):

    def __init__(self, *args: Any, num_patches: int=256, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.register_buffer('attn_mask', torch.ones(1, num_patches, num_patches, dtype=torch.bool).tril(diagonal=0))

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, **kwargs: Any) ->torch.Tensor:
        assert mask is not None, 'A mask is required for the PrefixLM Causal Attention.'
        B, N, C = x.shape
        prefix_mask = (~mask).unsqueeze(1).expand(-1, N, -1).bool()
        attn_mask = self.attn_mask.clone().expand(B, -1, -1)
        attn_mask = torch.logical_or(attn_mask, prefix_mask)
        attn_mask = attn_mask.unsqueeze(1)
        return super().forward(x, mask=attn_mask, **kwargs)


class LoraAttention(Attention):

    def __init__(self, dim: 'int', *args: Any, lora_rank: int=8, **kwargs: Any):
        super().__init__(dim, *args, **kwargs)
        self.qkv = loralib.MergedLinear(dim, dim * 3, bias=kwargs.get('qkv_bias', False), r=lora_rank, enable_lora=[True, False, True])
        self.proj = loralib.Linear(dim, dim, bias=kwargs.get('use_bias', True), r=lora_rank)


def _get_attention_target(dim: 'int', num_heads: 'int') ->Callable[[bool], nn.Module]:

    def callback(use_bias: 'bool') ->nn.Module:
        return layers.Attention(dim=dim, num_heads=num_heads, use_bias=use_bias)
    return callback


def _aim(img_size: 'Union[int, Tuple[int, int]]', patch_size: 'Union[int, Tuple[int, int]]', embed_dim: 'int', num_blocks: 'int', num_heads: 'int', num_channels: 'int'=3, probe_layers: 'Union[int, Tuple[int, ...]]'=6, num_classes: 'int'=1000, **kwargs: Any) ->Tuple[nn.Module, nn.Module, nn.Module]:
    norm_layer = layers.LayerNorm
    patchifier = layers.PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=num_channels, embed_dim=embed_dim, norm_layer=norm_layer)
    preprocessor = layers.ViTPreprocessor(patchifier, drop_patches=False, cls_token=False)
    if isinstance(probe_layers, int):
        probe_layers = tuple(range(num_blocks - probe_layers, num_blocks))
    assert all(layer >= 0 for layer in probe_layers), probe_layers
    attn_target = _get_attention_target(dim=embed_dim, num_heads=num_heads)
    post_transform_layer = layers.AverageLayers(probe_layers, reduce=False)
    trunk = Transformer(attn_target, embed_dim=embed_dim, num_blocks=num_blocks, norm_layer=norm_layer, post_transformer_layer=post_transform_layer, **kwargs)
    head = layers.AttentionPoolingClassifier(dim=embed_dim, out_features=num_classes, num_heads=num_heads, qkv_bias=False, num_queries=1)
    return preprocessor, trunk, head


class TextPreprocessor(nn.Module):

    def __init__(self, vocab_size: 'int', embed_dim: 'int', max_context_length: 'int'=77, eos_token_id: 'int'=49407):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(max_context_length, embed_dim))
        self.max_context_length = max_context_length
        self.eos_token_id = eos_token_id

    def forward(self, input_ids: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        _, N = input_ids.shape
        max_len = min(N, self.max_context_length)
        eos_token_mask = input_ids == self.eos_token_id
        tokens = self.text_embedding(input_ids)
        tokens = tokens[:, :max_len] + self.positional_embedding[:max_len].unsqueeze(0)
        return tokens, eos_token_mask


class ExtractEOS(nn.Module):

    def forward(self, tokens: 'torch.Tensor', eos_token_mask: 'torch.Tensor') ->torch.Tensor:
        B, _, D = tokens.shape
        eos_token_mask = torch.argmax(eos_token_mask.float(), dim=-1)
        assert eos_token_mask.shape == (B,)
        eos_token_mask = eos_token_mask.reshape(B, 1, 1).expand(B, 1, D)
        eos_token = torch.gather(tokens, 1, eos_token_mask)
        eos_token = eos_token.squeeze(1)
        return eos_token


RMSNorm = nn.RMSNorm


class SwiGLUFFN(nn.Module):

    def __init__(self, in_features: 'int', hidden_features: 'int', use_bias: 'bool'=True, norm_layer: 'Optional[Callable[[int], nn.Module]]'=None, **_: Any):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=use_bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.norm_layer = norm_layer(hidden_features) if norm_layer else nn.Identity()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.norm_layer(x)
        x = self.fc2(x)
        return x


class AIMv2LiT(nn.Module):

    def __init__(self, img_size: 'Union[int, Tuple[int, int]]'=224, patch_size: 'Union[int, Tuple[int, int]]'=14, projection_dim: 'int'=768, vision_embed_dim: 'int'=1024, vision_mlp_hidden_dim: 'int'=2816, vision_num_blocks: 'int'=24, vision_num_heads: 'int'=8, text_embed_dim: 'int'=768, text_mlp_embed_dim: 'int'=2048, text_num_blocks: 'int'=12, text_num_heads: 'int'=6, vocab_size: 'int'=49408, max_context_length: 'int'=77, eos_token_id: 'int'=49407, init_temperature: 'float'=0.07, max_logit_scale: 'float'=100.0, **kwargs: Any):
        super().__init__()
        self.image_encoder = AIMv2VisionEncoder(img_size=img_size, patch_size=patch_size, embed_dim=vision_embed_dim, mlp_hidden_dim=vision_mlp_hidden_dim, num_blocks=vision_num_blocks, num_heads=vision_num_heads, pos_embed_type='absolute', head_type='attention-pool', head_num_heads=vision_num_heads, head_num_queries=1, head_linear_bias=True, head_average_pool=True, **kwargs)
        self.text_encoder = AIMv2TextEncoder(embed_dim=text_embed_dim, mlp_hidden_dim=text_mlp_embed_dim, num_blocks=text_num_blocks, num_heads=text_num_heads, vocab_size=vocab_size, eos_token_id=eos_token_id, max_context_length=max_context_length)
        self.image_projector = nn.Linear(vision_embed_dim, projection_dim, bias=False)
        self.text_projector = nn.Linear(text_embed_dim, projection_dim, bias=False)
        self.log_logit_scale = nn.Parameter(torch.full([], fill_value=math.log(1.0 / init_temperature)))
        self.max_log_logit_scale = math.log(max_logit_scale)

    def forward(self, input_pixels: 'torch.Tensor', input_ids: 'torch.Tensor', output_features: 'bool'=False) ->Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        img_embeds = self.encode_image(input_pixels, output_features=False)
        img_embeds = F.normalize(img_embeds, p=2, dim=-1)
        text_embeds = self.encode_text(input_ids, output_features=False)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        logit_scale = self.log_logit_scale.clamp(0.0, self.max_log_logit_scale).exp()
        logits_per_text = logit_scale * text_embeds @ img_embeds.t()
        logits_per_image = logits_per_text.t()
        if output_features:
            return logits_per_image, logits_per_text, img_embeds, text_embeds
        return logits_per_image, logits_per_text

    def encode_image(self, input_pixels: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, output_features: 'bool'=False) ->Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        out = self.image_encoder(input_pixels, mask=mask, output_features=output_features)
        out = self.image_projector(out)
        return out

    def encode_text(self, input_ids: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, output_features: 'bool'=False) ->Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        out = self.text_encoder(input_ids, mask=mask, output_features=output_features)
        out = self.text_projector(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ExtractEOS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SwiGLUFFN,
     lambda: ([], {'in_features': 4, 'hidden_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_apple_ml_aim(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

