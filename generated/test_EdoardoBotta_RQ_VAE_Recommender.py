import sys
_module = sys.modules[__name__]
del sys
data = _module
amazon = _module
ml1m = _module
ml32m = _module
preprocessing = _module
processed = _module
schemas = _module
utils = _module
gumbel = _module
metrics = _module
kmeans = _module
modules = _module
id_embedder = _module
encoder = _module
loss = _module
model = _module
normalize = _module
quantize = _module
rqvae = _module
inv_sqrt = _module
semids = _module
attention = _module
model = _module
utils = _module
train_decoder = _module
train_rqvae = _module

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


import pandas as pd


import torch


from typing import Callable


from typing import List


from typing import Optional


import numpy as np


from enum import Enum


from torch.utils.data import Dataset


from typing import NamedTuple


from torch import Tensor


import torch.nn.functional as F


from typing import Tuple


from collections import defaultdict


from torch import nn


from torch.nn import functional as F


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LRScheduler


from torch.utils.data import BatchSampler


from torch.utils.data import DataLoader


from torch.utils.data import SequentialSampler


from torch.nested import Tensor as NestedTensor


from typing import Union


from torch.optim import AdamW


from torch.utils.data import RandomSampler


from torch.optim.lr_scheduler import ExponentialLR


class SemIdEmbedder(nn.Module):

    def __init__(self, num_embeddings, sem_ids_dim, embeddings_dim) ->None:
        super().__init__()
        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = sem_ids_dim * num_embeddings
        self.emb = nn.Embedding(num_embeddings=num_embeddings * self.sem_ids_dim + 1, embedding_dim=embeddings_dim, padding_idx=self.padding_idx)

    def forward(self, batch: 'TokenizedSeqBatch') ->Tensor:
        sem_ids = batch.token_type_ids * self.num_embeddings + batch.sem_ids
        sem_ids[~batch.seq_mask] = self.padding_idx
        return self.emb(sem_ids)


class UserIdEmbedder(nn.Module):

    def __init__(self, num_buckets, embedding_dim) ->None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)

    def forward(self, x: 'Tensor') ->Tensor:
        hashed_indices = x % self.num_buckets
        return self.emb(hashed_indices)


def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)


class L2NormalizationLayer(nn.Module):

    def __init__(self, dim=-1, eps=1e-12) ->None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x) ->Tensor:
        return l2norm(x, dim=self.dim, eps=self.eps)


class MLP(nn.Module):

    def __init__(self, input_dim: 'int', hidden_dims: 'List[int]', out_dim: 'int', normalize: 'bool'=False) ->None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]
        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i != len(dims) - 2:
                self.mlp.append(nn.SiLU())
        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())

    def forward(self, x: 'Tensor') ->Tensor:
        assert x.shape[-1] == self.input_dim, f'Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}'
        return self.mlp(x)


class ReconstructionLoss(nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def forward(self, x_hat: 'Tensor', x: 'Tensor') ->Tensor:
        return ((x_hat - x) ** 2).sum(axis=-1)


class CategoricalReconstuctionLoss(nn.Module):

    def __init__(self, n_cat_feats: 'int') ->None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats

    def forward(self, x_hat: 'Tensor', x: 'Tensor') ->Tensor:
        reconstr = self.reconstruction_loss(x_hat[:, :-self.n_cat_feats], x[:, :-self.n_cat_feats])
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(x_hat[:, -self.n_cat_feats:], x[:, -self.n_cat_feats:], reduction='none').sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class QuantizeLoss(nn.Module):

    def __init__(self, commitment_weight: 'float'=1.0) ->None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: 'Tensor', value: 'Tensor') ->Tensor:
        emb_loss = ((query.detach() - value) ** 2).sum(axis=[-1])
        query_loss = ((query - value.detach()) ** 2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss


AttentionInput = Union[Tensor, NestedTensor]


class GenerationOutput(NamedTuple):
    sem_ids: 'Tensor'
    log_probas: 'Tensor'


class ModelOutput(NamedTuple):
    loss: 'Tensor'
    logits: 'Tensor'


class TokenizedSeqBatch(NamedTuple):
    user_ids: 'Tensor'
    sem_ids: 'Tensor'
    sem_ids_fut: 'Tensor'
    seq_mask: 'Tensor'
    token_type_ids: 'Tensor'


class Attend(nn.Module):

    def __init__(self, d_out, num_heads, head_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = d_out
        self.dropout = dropout

    def jagged_forward(self, qu: 'NestedTensor', ke: 'NestedTensor', va: 'NestedTensor', is_causal: 'bool') ->NestedTensor:
        queries = qu.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        keys = ke.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        values = va.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        dropout_p = 0.0 if not self.training else self.dropout
        context_vec = F.scaled_dot_product_attention(queries, keys, values, dropout_p=dropout_p, is_causal=is_causal)
        context_vec = context_vec.transpose(1, 2).flatten(-2)
        return context_vec

    def forward(self, qkv: 'Tensor', attn_mask: 'Tensor') ->Tensor:
        batch_size, num_tokens, embed_dim = qkv.shape
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv
        use_dropout = 0.0 if not self.training else self.dropout
        context_vec = F.scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=attn_mask is None)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return context_vec


@torch.compiler.disable
def padded_to_jagged_tensor(x: 'Tensor', lengths: 'Tensor') ->NestedTensor:
    return torch.nested.nested_tensor([i[:j.item()] for i, j in zip(x, lengths)], layout=torch.jagged, device=x.device)


class KVCache(nn.Module):

    def __init__(self, dim):
        super().__init__()
        assert len(dim) == 3, 'Cache only supports 3d tensors'
        self.register_buffer('k_cache', torch.zeros(*dim, requires_grad=False))
        self.register_buffer('v_cache', torch.zeros(*dim, requires_grad=False))
        self.dim = dim
        self._reset_limits()
        self.is_empty = True

    def _reset_limits(self):
        self.cache_limits = [(0) for _ in self.dim]
        self.next_seq_pos = None

    def reset(self):
        self.k_cache.fill_(0)
        self.v_cache.fill_(0)
        self._reset_limits()
        self.is_empty = True

    @property
    def device(self):
        return self.k_cache.device

    @property
    def keys(self):
        B, N, D = self.cache_limits
        return self.k_cache[:B, :N, :D]

    @property
    def values(self):
        B, N, D = self.cache_limits
        return self.v_cache[:B, :N, :D]

    @property
    def seq_lengths(self):
        if self.is_empty:
            return 0
        return self.next_seq_pos

    @torch.no_grad
    def store(self, keys: 'Tensor', values: 'Tensor', mask: 'Tensor') ->None:
        B, N = mask.shape
        self.k_cache[:B, :N, :][mask] = keys.detach()[:, :]
        self.v_cache[:B, :N, :][mask] = values.detach()[:, :]
        self.cache_limits = [B, N, self.dim[-1]]
        self.next_seq_pos = mask.sum(axis=1).unsqueeze(-1)
        self.is_empty = False

    @torch.no_grad
    def append_column(self, keys: 'Tensor', values: 'Tensor') ->None:
        B, N, D = self.cache_limits
        row_idx = torch.arange(B, device=self.k_cache.device)
        self.k_cache[:B, :][row_idx, self.next_seq_pos] = keys.detach()[:, :]
        self.v_cache[:B, :][row_idx, self.next_seq_pos] = values.detach()[:, :]
        max_pos_appended = self.next_seq_pos.max()
        if max_pos_appended >= N:
            self.cache_limits[1] = max_pos_appended + 1
        self.next_seq_pos += 1

    @torch.no_grad
    @torch.compiler.disable
    def as_jagged(self):
        keys_jagged = padded_to_jagged_tensor(self.keys, self.seq_lengths.squeeze())
        values_jagged = padded_to_jagged_tensor(self.values, self.seq_lengths.squeeze())
        return keys_jagged, values_jagged

    @torch.no_grad
    def apply(self, fn) ->None:
        B, N, D = self.cache_limits
        k_transformed, v_transformed = fn(self.k_cache[:B, :N, :D]), fn(self.v_cache[:B, :N, :D])
        next_seq_pos_transformed = fn(self.next_seq_pos)
        B, N, D = k_transformed.shape
        self.reset()
        self.k_cache[:B, :N, :D] = k_transformed
        self.v_cache[:B, :N, :D] = v_transformed
        self.next_seq_pos = next_seq_pos_transformed
        self.cache_limits = [B, N, D]
        self.is_empty = False


def jagged_to_flattened_tensor(x: 'NestedTensor') ->Tensor:
    return x.values()


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, num_heads, cross_attn=False, dropout=0.0, qkv_bias=False) ->None:
        super().__init__()
        assert d_out % num_heads == 0, 'embed_dim is indivisible by num_heads'
        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        if self.cross_attn:
            self.q = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.kv = nn.Linear(d_in, 2 * d_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.attend = Attend(self.d_out, self.num_heads, self.head_dim, self.dropout)
        self._kv_cache = KVCache((2560, 80, 384))

    @property
    def kv_cache(self) ->KVCache:
        return self._kv_cache

    def forward(self, x: 'AttentionInput', x_kv: 'Optional[AttentionInput]'=None, padding_mask: 'Optional[Tensor]'=None, attn_mask: 'Optional[Tensor]'=None, jagged: 'bool'=False, use_cache: 'bool'=False) ->AttentionInput:
        assert not self.cross_attn or x_kv is not None, 'Found null x_kv in cross attn. layer'
        if self.cross_attn:
            q, kv = self.q(x), self.kv(x_kv)
            qkv = torch.cat([q, kv], axis=2)
        else:
            qkv = self.qkv(x)
        if not self.training and use_cache and self.kv_cache.is_empty:
            assert padding_mask is not None
            B, N = padding_mask.shape
            queries, keys, values = qkv.chunk(3, dim=-1)
            self.kv_cache.store(keys=jagged_to_flattened_tensor(keys), values=jagged_to_flattened_tensor(values), mask=padding_mask)
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=True)
        elif not self.training and use_cache and not self.kv_cache.is_empty:
            assert padding_mask is not None
            B, N = padding_mask.shape
            queries, keys, values = qkv.chunk(3, dim=-1)
            keys, values = jagged_to_flattened_tensor(keys), jagged_to_flattened_tensor(values)
            self.kv_cache.append_column(keys=keys, values=values)
            keys, values = self.kv_cache.as_jagged()
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=False)
        elif jagged:
            queries, keys, values = torch.chunk(qkv, 3, dim=-1)
            context_vec = self.attend.jagged_forward(queries, keys, values, is_causal=True)
        if not jagged:
            context_vec = self.attend(qkv, attn_mask)
        context_vec = self.proj(context_vec)
        return context_vec


class RMSNorm(nn.Module):

    def __init__(self, dim: 'int', eps: 'float'=1e-06) ->None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) ->Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):

    def __init__(self, d_in: 'int', d_out: 'int', dropout: 'float', num_heads: 'int', qkv_bias: 'bool', mlp_hidden_dims: 'List[int]'=[1024], do_cross_attn: 'bool'=False) ->None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = dropout
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.do_cross_attn = do_cross_attn
        self.attention = MultiHeadAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, cross_attn=False, dropout=dropout, qkv_bias=qkv_bias)
        self.ff = MLP(input_dim=d_out, hidden_dims=mlp_hidden_dims, out_dim=d_out, normalize=False)
        self.attn_norm = RMSNorm(d_out)
        self.ffn_norm = RMSNorm(d_out)
        if self.do_cross_attn:
            self.cross_attention = MultiHeadAttention(d_in=d_out, d_out=d_out, num_heads=num_heads, cross_attn=True, dropout=dropout, qkv_bias=qkv_bias)
            self.cross_attn_norm = RMSNorm(d_out)

    def forward(self, x: 'AttentionInput', x_kv: 'Optional[Tensor]'=None, padding_mask: 'Optional[Tensor]'=None, attn_mask: 'Optional[Tensor]'=None, jagged: 'Optional[bool]'=False) ->AttentionInput:
        attn_out = self.attn_norm(x + self.attention(x, padding_mask=padding_mask, attn_mask=attn_mask, jagged=jagged, use_cache=not self.training))
        if self.do_cross_attn:
            attn_out = self.cross_attn_norm(attn_out + self.cross_attention(x_q=attn_out, x_kv=x_kv, padding_mask=padding_mask, attn_mask=attn_mask, jagged=jagged, use_cache=not self.training))
        proj_out = self.ffn_norm(attn_out + self.ff(attn_out))
        return proj_out

    def reset_kv_cache(self):
        self.attention.kv_cache.reset()
        if self.do_cross_attn:
            self.cross_attention.kv_cache.reset()

    def apply_to_kv_cache(self, fn):
        self.attention.kv_cache.apply(fn)
        if self.do_cross_attn:
            self.cross_attention.kv_cache.apply(fn)


class TransformerDecoder(nn.Module):

    def __init__(self, d_in: 'int', d_out: 'int', dropout: 'float', num_heads: 'int', n_layers: 'int', do_cross_attn: 'bool'=False) ->None:
        super().__init__()
        self.do_cross_attn = do_cross_attn
        self.layers = nn.ModuleList([TransformerBlock(d_in=d_in, d_out=d_out, dropout=dropout, num_heads=num_heads, qkv_bias=False, do_cross_attn=self.do_cross_attn) for _ in range(n_layers)])

    def forward(self, x: 'AttentionInput', padding_mask: 'Optional[Tensor]'=None, attn_mask: 'Optional[Tensor]'=None, context: 'Optional[Tensor]'=None, jagged: 'Optional[bool]'=None) ->AttentionInput:
        for layer in self.layers:
            x = layer(x=x, x_kv=context, padding_mask=padding_mask, attn_mask=attn_mask, jagged=jagged)
        return x

    @property
    def seq_lengths(self) ->Tensor:
        return self.layers[0].attention.kv_cache.seq_lengths

    def reset_kv_cache(self) ->None:
        for layer in self.layers:
            layer.reset_kv_cache()

    def apply_to_kv_cache(self, fn) ->None:
        for layer in self.layers:
            layer.apply_to_kv_cache(fn)


def eval_mode(fn):

    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner


def maybe_repeat_interleave(x, repeats, dim):
    if not isinstance(x, Tensor):
        return x
    return x.repeat_interleave(repeats, dim=dim)


def reset_kv_cache(fn):

    def inner(self, *args, **kwargs):
        self.decoder.reset_kv_cache()
        out = fn(self, *args, **kwargs)
        self.decoder.reset_kv_cache()
        return out
    return inner


def select_columns_per_row(x: 'Tensor', indices: 'Tensor') ->torch.Tensor:
    assert x.shape[0] == indices.shape[0]
    assert indices.shape[1] <= x.shape[1]
    B = x.shape[0]
    return x[rearrange(torch.arange(B, device=x.device), 'B -> B 1'), indices]


class DecoderRetrievalModel(nn.Module):

    def __init__(self, embedding_dim, attn_dim, dropout, num_heads, n_layers, num_embeddings, sem_id_dim, inference_verifier_fn, max_pos=2048, jagged_mode: 'bool'=True) ->None:
        super().__init__()
        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.sem_id_embedder = SemIdEmbedder(num_embeddings=num_embeddings, sem_ids_dim=sem_id_dim, embeddings_dim=embedding_dim)
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.decoder = TransformerDecoder(d_in=attn_dim, d_out=attn_dim, dropout=dropout, num_heads=num_heads, n_layers=n_layers, do_cross_attn=False)
        self.in_proj = nn.Linear(embedding_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)

    def _predict(self, batch: 'TokenizedSeqBatch') ->AttentionInput:
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        B, N, D = sem_ids_emb.shape
        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0) + self.decoder.seq_lengths
        wpe = self.wpe(pos)
        tte = self.tte(batch.token_type_ids)
        input_embedding = wpe + sem_ids_emb + tte
        if self.jagged_mode:
            seq_lengths = batch.seq_mask.sum(axis=1)
            input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths)
        transformer_input = self.in_proj(input_embedding)
        transformer_output = self.decoder(transformer_input, padding_mask=batch.seq_mask, jagged=self.jagged_mode)
        return transformer_output

    @eval_mode
    @reset_kv_cache
    @torch.no_grad
    def generate_next_sem_id(self, batch: 'TokenizedSeqBatch', temperature: 'int'=1, top_k: 'bool'=True) ->GenerationOutput:
        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 10 if top_k else 1
        n_top_k_candidates = 2 * k if top_k else 1
        next_token_pos = batch.seq_mask.sum(axis=1)
        to_shift = next_token_pos > N - self.sem_id_dim
        batch.sem_ids[to_shift, :] = batch.sem_ids[to_shift].roll(-self.sem_id_dim, dims=1)
        batch.sem_ids[to_shift, N - self.sem_id_dim:] = -1
        batch.seq_mask[to_shift, N - self.sem_id_dim:] = False
        next_token_pos = batch.seq_mask.sum(axis=1).repeat_interleave(k)
        for _ in range(self.sem_id_dim):
            logits = self.forward(batch).logits
            probas = F.softmax(logits / temperature, dim=-1)
            samples = torch.multinomial(probas, num_samples=n_top_k_candidates)
            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(samples.unsqueeze(-1))
            else:
                prefix = torch.cat([generated.flatten(0, 1).unsqueeze(1).repeat_interleave(n_top_k_candidates, axis=1), samples.unsqueeze(-1)], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)
            samples = samples.reshape(B, -1)
            probas = probas.reshape(B, -1)
            sampled_log_probas = torch.log(select_columns_per_row(probas, samples))
            sorted_log_probas, sorted_indices = (-10000 * ~is_valid_prefix + sampled_log_probas + maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)).sort(-1, descending=True)
            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = select_columns_per_row(samples, top_k_indices)
            if generated is not None:
                parent_id = select_columns_per_row(generated, top_k_indices // n_top_k_candidates)
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)
                cache_idx = (torch.arange(B, device=top_k_indices.device).unsqueeze(-1) * k + top_k_indices // n_top_k_candidates).flatten()
                self.decoder.apply_to_kv_cache(lambda x: x[cache_idx])
                batch = TokenizedSeqBatch(user_ids=batch.user_ids, sem_ids=top_k_samples[:, :, -1].reshape(-1, 1), sem_ids_fut=batch.sem_ids_fut, seq_mask=batch.seq_mask, token_type_ids=batch.token_type_ids + 1)
                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)
                next_batch_size = next_sem_ids.shape[0]
                batch = TokenizedSeqBatch(user_ids=batch.user_ids.repeat_interleave(k, dim=0), sem_ids=next_sem_ids, sem_ids_fut=batch.sem_ids_fut, seq_mask=torch.ones(next_batch_size, 1, dtype=bool, device=next_sem_ids.device), token_type_ids=torch.zeros(next_batch_size, 1, dtype=torch.int32, device=next_sem_ids.device))
                self.decoder.apply_to_kv_cache(lambda x: x.repeat_interleave(k, axis=0))
                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())
        return GenerationOutput(sem_ids=generated.squeeze(), log_probas=log_probas.squeeze())

    @torch.compile
    def forward(self, batch: 'TokenizedSeqBatch') ->ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape
        trnsf_out = self._predict(batch)
        if self.training:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                logits = jagged_to_flattened_tensor(predict_out)
                target = torch.cat([batch.sem_ids[:, 1:], -torch.ones(B, 1, device=batch.sem_ids.device, dtype=batch.sem_ids.dtype)], axis=1)[seq_mask]
                loss = F.cross_entropy(logits, target, ignore_index=-1)
            else:
                logits = predict_out
                target_mask = seq_mask[:, 1:]
                out = logits[:, :-1, :][target_mask, :]
                target = batch.sem_ids[:, 1:][target_mask]
                loss = F.cross_entropy(out, target)
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            last_token_pos = trnsf_out.offsets()[1:] - 1
            trnsf_out_flattened = jagged_to_flattened_tensor(trnsf_out)
            logits = self.out_proj(trnsf_out_flattened[last_token_pos, :])
            loss = None
        else:
            last_token_pos = seq_mask.sum(axis=-1) - 1
            logits = self.out_proj(trnsf_out[torch.arange(B, device=trnsf_out.device), last_token_pos])
            loss = None
        return ModelOutput(loss=loss, logits=logits)


class QuantizeOutput(NamedTuple):
    embeddings: 'Tensor'
    ids: 'Tensor'
    loss: 'Tensor'


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = F.normalize(u + q, p=2, dim=1, eps=1e-06).detach()
    return (e - 2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) + 2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())).squeeze()


def sample_gumbel(shape: 'Tuple', device: 'torch.device', eps=1e-20) ->Tensor:
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: 'Tensor', temperature: 'float', device: 'torch.device') ->Tensor:
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample


class KmeansOutput(NamedTuple):
    centroids: 'torch.Tensor'
    assignment: 'torch.Tensor'


class Kmeans:

    def __init__(self, k: 'int', max_iters: 'int'=None, stop_threshold: 'float'=1e-10) ->None:
        self.k = k
        self.iters = max_iters
        self.stop_threshold = stop_threshold
        self.centroids = None
        self.assignment = None

    def _init_centroids(self, x: 'torch.Tensor') ->None:
        B, D = x.shape
        init_idx = np.random.choice(B, self.k, replace=False)
        self.centroids = x[init_idx, :]
        self.assignment = None

    def _update_centroids(self, x) ->torch.Tensor:
        squared_pw_dist = (rearrange(x, 'b d -> b 1 d') - rearrange(self.centroids, 'b d -> 1 b d')) ** 2
        centroid_idx = squared_pw_dist.sum(axis=2).min(axis=1).indices
        assigned = rearrange(torch.arange(self.k, device=x.device), 'd -> d 1') == centroid_idx
        for cluster in range(self.k):
            is_assigned_to_c = assigned[cluster]
            if not is_assigned_to_c.any():
                if x.size(0) > 0:
                    self.centroids[cluster, :] = x[torch.randint(0, x.size(0), (1,))].squeeze(0)
                else:
                    raise ValueError('Can not choose random element from x, x is empty')
            else:
                self.centroids[cluster, :] = x[is_assigned_to_c, :].mean(axis=0)
        self.assignment = centroid_idx

    def run(self, x):
        self._init_centroids(x)
        i = 0
        while self.iters is None or i < self.iters:
            old_c = self.centroids.clone()
            self._update_centroids(x)
            if torch.norm(self.centroids - old_c, dim=1).max() < self.stop_threshold:
                break
            i += 1
        return KmeansOutput(centroids=self.centroids, assignment=self.assignment)


def kmeans_init_(tensor: 'torch.Tensor', x: 'torch.Tensor'):
    assert tensor.dim() == 2
    assert x.dim() == 2
    with torch.no_grad():
        k, _ = tensor.shape
        kmeans_out = Kmeans(k=k).run(x)
        tensor.data.copy_(kmeans_out.centroids)


class RqVaeComputedLosses(NamedTuple):
    loss: 'Tensor'
    reconstruction_loss: 'Tensor'
    rqvae_loss: 'Tensor'
    embs_norm: 'Tensor'
    p_unique_ids: 'Tensor'


class RqVaeOutput(NamedTuple):
    embeddings: 'Tensor'
    residuals: 'Tensor'
    sem_ids: 'Tensor'
    quantize_loss: 'Tensor'


class SeqBatch(NamedTuple):
    user_ids: 'Tensor'
    ids: 'Tensor'
    ids_fut: 'Tensor'
    x: 'Tensor'
    x_fut: 'Tensor'
    seq_mask: 'Tensor'


def batch_to(batch, device):
    return SeqBatch(*[v for _, v in batch._asdict().items()])


class SemanticIdTokenizer(nn.Module):
    """
        Tokenizes a batch of sequences of item features into a batch of sequences of semantic ids.
    """

    def __init__(self, input_dim: 'int', output_dim: 'int', hidden_dims: 'List[int]', codebook_size: 'int', n_layers: 'int'=3, n_cat_feats: 'int'=18, commitment_weight: 'float'=0.25, rqvae_weights_path: 'Optional[str]'=None, rqvae_codebook_normalize: 'bool'=False, rqvae_sim_vq: 'bool'=False) ->None:
        super().__init__()
        self.rq_vae = RqVae(input_dim=input_dim, embed_dim=output_dim, hidden_dims=hidden_dims, codebook_size=codebook_size, codebook_kmeans_init=False, codebook_normalize=rqvae_codebook_normalize, codebook_sim_vq=rqvae_sim_vq, n_layers=n_layers, n_cat_features=n_cat_feats, commitment_weight=commitment_weight)
        if rqvae_weights_path is not None:
            self.rq_vae.load_pretrained(rqvae_weights_path)
        self.rq_vae.eval()
        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.reset()

    def _get_hits(self, query: 'Tensor', key: 'Tensor') ->Tensor:
        return (rearrange(key, 'b d -> 1 b d') == rearrange(query, 'b d -> b 1 d')).all(axis=-1)

    def reset(self):
        self.cached_ids = None

    @property
    def sem_ids_dim(self):
        return self.n_layers + 1

    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: 'ItemData') ->Tensor:
        cached_ids = None
        dedup_dim = []
        sampler = BatchSampler(SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False)
        dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, collate_fn=lambda batch: batch[0])
        for batch in dataloader:
            batch_ids = self.forward(batch_to(batch, self.rq_vae.device)).sem_ids
            is_hit = self._get_hits(batch_ids, batch_ids)
            hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
            assert hits.min() >= 0
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                is_hit = self._get_hits(batch_ids, cached_ids)
                hits += is_hit.sum(axis=-1)
                cached_ids = pack([cached_ids, batch_ids], '* d')[0]
            dedup_dim.append(hits)
        dedup_dim_tensor = pack(dedup_dim, '*')[0]
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], 'b *')[0]
        return self.cached_ids

    @torch.no_grad
    @eval_mode
    def exists_prefix(self, sem_id_prefix: 'Tensor') ->Tensor:
        if self.cached_ids is None:
            raise Exception('No match can be found in empty cache.')
        prefix_length = sem_id_prefix.shape[-1]
        prefix_cache = self.cached_ids[:, :prefix_length]
        return (sem_id_prefix.unsqueeze(-2) == prefix_cache.unsqueeze(-3)).all(axis=-1).any(axis=-1)

    def _tokenize_seq_batch_from_cached(self, ids: 'Tensor') ->Tensor:
        return rearrange(self.cached_ids[ids.flatten(), :], '(b n) d -> b (n d)', n=ids.shape[1])

    @torch.no_grad
    @eval_mode
    def forward(self, batch: 'SeqBatch') ->TokenizedSeqBatch:
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            sem_ids = self.rq_vae.get_semantic_ids(batch.x).sem_ids
            D = sem_ids.shape[-1]
            seq_mask, sem_ids_fut = None, None
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1
            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        return TokenizedSeqBatch(user_ids=batch.user_ids, sem_ids=sem_ids, sem_ids_fut=sem_ids_fut, seq_mask=seq_mask, token_type_ids=token_type_ids)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CategoricalReconstuctionLoss,
     lambda: ([], {'n_cat_feats': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (L2NormalizationLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'d_in': 4, 'd_out': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (QuantizeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReconstructionLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_EdoardoBotta_RQ_VAE_Recommender(_paritybench_base):
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

