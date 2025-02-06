import sys
_module = sys.modules[__name__]
del sys
bert = _module
preprocess = _module
model = _module
task = _module
setup = _module
test_softmax = _module
unicore = _module
checkpoint_utils = _module
data = _module
append_token_dataset = _module
base_wrapper_dataset = _module
bert_tokenize_dataset = _module
data_utils = _module
dictionary = _module
from_numpy_dataset = _module
iterators = _module
lmdb_dataset = _module
lru_cache_dataset = _module
mask_tokens_dataset = _module
nested_dictionary_dataset = _module
num_samples_dataset = _module
numel_dataset = _module
pad_dataset = _module
prepend_token_dataset = _module
raw_dataset = _module
sort_dataset = _module
tokenize_dataset = _module
unicore_dataset = _module
distributed = _module
legacy_distributed_data_parallel = _module
module_proxy_wrapper = _module
utils = _module
ema = _module
logging = _module
meters = _module
metrics = _module
progress_bar = _module
losses = _module
cross_entropy = _module
masked_lm = _module
unicore_loss = _module
models = _module
distributed_unicore_model = _module
unicore_model = _module
modules = _module
layer_norm = _module
multihead_attention = _module
softmax_dropout = _module
transformer_decoder = _module
transformer_decoder_layer = _module
transformer_encoder = _module
transformer_encoder_layer = _module
nan_detector = _module
optim = _module
adadelta = _module
adagrad = _module
adam = _module
dynamic_loss_scaler = _module
fp16_optimizer = _module
fused_adam = _module
lr_scheduler = _module
cosine_lr_scheduler = _module
exponential_decay_schedule = _module
fixed_schedule = _module
inverse_square_root_schedule = _module
pass_through = _module
polynomial_decay_schedule = _module
reduce_lr_on_plateau = _module
tri_stage_lr_scheduler = _module
triangular_lr_scheduler = _module
unicore_lr_scheduler = _module
sgd = _module
unicore_optimizer = _module
options = _module
registry = _module
tasks = _module
unicore_task = _module
trainer = _module
utils = _module
version = _module
unicore_cli = _module
train = _module

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


import logging


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.utils import cpp_extension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import BuildExtension


import collections


import re


from typing import Any


from typing import Dict


from typing import Optional


import numpy as np


from functools import lru_cache


from torch.utils.data.dataloader import default_collate


import itertools


import math


import queue


import time


from collections import OrderedDict


import torch.utils.data


from torch import nn


import random


import warnings


from typing import List


from typing import Mapping


import torch.distributed as dist


from copy import deepcopy


from itertools import chain


import uuid


from collections import defaultdict


from typing import Callable


from numbers import Number


import inspect


from torch.nn.modules.loss import _Loss


from torch.nn.parallel import DistributedDataParallel


import numbers


from torch.nn.parameter import Parameter


from torch.nn import init


from torch.nn import functional as F


from torch import Tensor


import torch.optim


from collections.abc import Collection


import torch.optim.lr_scheduler


from functools import partial


import torch.utils.checkpoint


from typing import Tuple


FUSED_LAYER_NORM_SUPPORT_DIM = set([64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 1792, 2048, 2560, 5120])


class FusedLayerNormFastFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input = input.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        output, mean, invvar = unicore_fused_layernorm.forward(input, ctx.normalized_shape, weight, bias, ctx.eps)
        ctx.save_for_backward(input, weight, bias, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input = unicore_fused_layernorm.backward(grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        grad_weight, grad_bias = unicore_fused_layernorm_backward_gamma_beta.backward(grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        return grad_input, grad_weight, grad_bias, None, None


class LayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

        def torch_layer_norm(input):
            return F.layer_norm(input, self.normalized_shape, self.weight.type(input.dtype), self.bias.type(input.dtype), self.eps)

        def fused_layer_norm(input):
            if input.is_cuda:
                return FusedLayerNormFastFunction.apply(input, self.weight.type(input.dtype), self.bias.type(input.dtype), self.normalized_shape, self.eps)
            else:
                return F.layer_norm(input, self.normalized_shape, self.weight.type(input.dtype), self.bias.type(input.dtype), self.eps)
        self.func = torch_layer_norm if not HAS_LAYER_NORM or normalized_shape[0] not in FUSED_LAYER_NORM_SUPPORT_DIM else fused_layer_norm

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):
        return self.func(input)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine=True'.format(**self.__dict__)


class BertLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class BertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes), q_noise, qn_block_size

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ModuleProxyWrapper(nn.Module):
    """
    Wrap a DistributedDataParallel module and forward requests for missing
    attributes to the module wrapped by DDP (the twice-wrapped module).
    Also forward calls to :func:`state_dict` and :func:`load_state_dict`.

    Usage::

        module.xyz = "hello world"
        wrapped_module = DistributedDataParallel(module, **ddp_args)
        wrapped_module = ModuleProxyWrapper(wrapped_module)
        assert wrapped_module.xyz == "hello world"
        assert wrapped_module.state_dict().keys() == module.state_dict().keys()

    Args:
        module (nn.Module): module to wrap
    """

    def __init__(self, module: 'nn.Module'):
        super().__init__()
        assert hasattr(module, 'module'), 'ModuleProxyWrapper expects input to wrap another module'
        self.module = module

    def __getattr__(self, name):
        """Forward missing attributes to twice-wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.module, name)
            except AttributeError:
                return getattr(self.module.module, name)

    def state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class UnicoreLoss(_Loss):

    def __init__(self, task):
        super().__init__()
        self.task = task
        if task is not None:
            self.args = task.args
            if hasattr(task, 'target_dictionary'):
                tgt_dict = task.target_dictionary
                self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100

    @classmethod
    def add_args(cls, parser):
        pass

    @classmethod
    def build_loss(cls, args, task):
        """Construct a loss from command-line args."""
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if p.kind == p.POSITIONAL_ONLY or p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD:
                raise NotImplementedError('{} not supported'.format(p.kind))
            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
            if p.name == 'task':
                init_args['task'] = task
            elif p.name == 'args':
                init_args['args'] = args
            elif hasattr(args, p.name):
                init_args[p.name] = getattr(args, p.name)
            elif p.default != p.empty:
                pass
            else:
                raise NotImplementedError('Unable to infer Loss arguments, please implement {}.build_loss'.format(cls.__name__))
        return cls(**init_args)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def logging_outputs_can_be_summed(is_train: 'bool') ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


class BaseUnicoreModel(nn.Module):
    """Base class for unicore models."""

    def __init__(self):
        super().__init__()

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError('Model must implement the build_model method')

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, model_args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. 
        """
        return super().load_state_dict(state_dict, strict)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, 'set_num_updates') and m != self:
                m.set_num_updates(num_updates)
        self.apply(_apply)


class SoftmaxDropoutFast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, is_training, inputs, mask, bias, dropout_prob):
        dropout_results, dropout_mask, softmax_results = unicore_fused_softmax_dropout.forward(is_training, inputs, mask, bias, dropout_prob, None)
        if is_training:
            ctx.dropout_prob = dropout_prob
            ctx.save_for_backward(softmax_results, dropout_mask)
            ctx.has_bias = bias is not None and bias.requires_grad
            if ctx.has_bias:
                ctx.bias_batch_dim = bias.shape[0]
        return dropout_results

    @staticmethod
    def backward(ctx, grad_output):
        softmax_results, dropout_mask = ctx.saved_tensors
        dropout_prob = ctx.dropout_prob
        grad_output = grad_output.contiguous()
        grad_input = unicore_fused_softmax_dropout.backward(grad_output, softmax_results, dropout_mask, dropout_prob)
        if ctx.has_bias:
            grad_bias = grad_input.view(-1, ctx.bias_batch_dim, grad_input.shape[-2], grad_input.shape[-1]).sum(dim=0)
        else:
            grad_bias = None
        return None, grad_input, None, grad_bias, None


def _check_bias(bias, input):
    try:
        assert bias.dtype == input.dtype, 'bias and input must have the same dtype'
        assert len(bias.shape) == len(input.shape), 'wrong length of bias.shape'
        assert bias.shape[-1] == input.shape[-1], 'bias.shape[-1] must be input.shape[-1]'
        assert bias.shape[-2] == input.shape[-2], 'bias.shape[-2] must be input.shape[-2]'
        len_shape = len(input.shape)
        if len_shape > 3:
            assert bias.shape[-3] == input.shape[-3], 'bias.shape[-3] must be input.shape[-3]'
            offset = 3
        else:
            offset = 2
        prev_non_one = True
        for i in range(len_shape - offset - 1, -1, -1):
            if prev_non_one:
                assert bias.shape[i] == input.shape[i] or bias.shape[i] == 1, 'bias.shape[{}] must be input.shape[{}] or 1'.format(i, i)
            else:
                assert bias.shape[i] == 1, 'bias.shape[{}] must be 1'.format(i)
            prev_non_one = bias.shape[i] != 1
        return True
    except:
        return False


def _check_mask(mask, input):
    try:
        assert mask.dtype == input.dtype, 'mask and input must have the same dtype'
        assert len(mask.shape) == len(input.shape), 'wrong length of mask.shape'
        assert mask.shape[-3] == 1 or mask.shape[-3] == input.shape[-3], 'mask.shape[-3] must be 1 or input.shape[-3]'
        if mask.shape[-3] == 1:
            assert mask.shape[-2] == 1, 'when mask.shape[-3] == 1, mask.shape[-2] must be 1'
        else:
            assert mask.shape[-2] == 1 or mask.shape[-2] == input.shape[-2], 'mask.shape[-2] must be 1 or input.shape[-2]'
        return True
    except:
        return False


def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None, inplace=True):
    """softmax dropout, and mask, bias are optional.
    Args:
        input (torch.Tensor): input tensor
        dropout_prob (float): dropout probability
        is_training (bool, optional): is in training or not. Defaults to True.
        mask (torch.Tensor, optional): the mask tensor, use as input + mask . Defaults to None.
        bias (torch.Tensor, optional): the bias tensor, use as input + bias . Defaults to None.

    Returns:
        torch.Tensor: the result after softmax
    """
    input = input.contiguous()
    if not inplace:
        input = input.clone()
    if input.is_cuda and HAS_SOFTMAX:
        input_size = input.size()
        if mask is not None:
            if _check_mask(mask, input):
                mask = mask.contiguous().view(-1, mask.shape[-2], mask.shape[-1])
            else:
                input += mask
                mask = None
        if bias is not None:
            if _check_bias(bias, input):
                bias = bias.contiguous().view(-1, input_size[-2], input_size[-1])
            else:
                input += bias
                bias = None
        input = input.view(-1, input_size[-2], input_size[-1])
        if dropout_prob <= 0.0 or input_size[-1] <= 1024:
            return SoftmaxDropoutFast.apply(is_training, input, mask, bias, dropout_prob).view(*input_size)
        else:
            return F.dropout(SoftmaxDropoutFast.apply(is_training, input, mask, bias, 0.0).view(*input_size), p=dropout_prob, training=is_training)
    else:
        if mask is not None:
            input += mask
        if bias is not None:
            input += bias
        return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)


class SelfMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, scaling_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = (self.head_dim * scaling_factor) ** -0.5
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key_padding_mask: 'Optional[Tensor]'=None, attn_bias: 'Optional[Tensor]'=None, return_attn: 'bool'=False) ->Tensor:
        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim) * self.scaling
        if k is not None:
            k = k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        if v is not None:
            v = v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        assert k is not None
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if not return_attn:
            attn = softmax_dropout(attn_weights, self.dropout, self.training, bias=attn_bias)
        else:
            attn_weights += attn_bias
            attn = softmax_dropout(attn_weights, self.dropout, self.training, inplace=False)
        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        o = o.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn


class CrossMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, scaling_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = (self.head_dim * scaling_factor) ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, key_padding_mask: 'Optional[Tensor]'=None, attn_bias: 'Optional[Tensor]'=None) ->Tensor:
        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim) * self.scaling
        if k is not None:
            k = k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        if v is not None:
            v = v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        assert k is not None
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn = softmax_dropout(attn_weights, self.dropout, self.training, bias=attn_bias)
        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        o = o.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        o = self.out_proj(o)
        return o


class TransformerDecoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, embed_dim: 'int'=768, ffn_embed_dim: 'int'=3072, attention_heads: 'int'=8, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.0, activation_fn: 'str'='gelu', post_ln=False) ->None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = SelfMultiheadAttention(self.embed_dim, attention_heads, dropout=attention_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = CrossMultiheadAttention(self.embed_dim, attention_heads, dropout=attention_dropout)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln

    def forward(self, x: 'torch.Tensor', encoder_out: 'torch.Tensor'=None, attn_bias: 'Optional[torch.Tensor]'=None, padding_mask: 'Optional[torch.Tensor]'=None, encoder_attn_bias: 'Optional[torch.Tensor]'=None, encoder_padding_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(query=x, key_padding_mask=padding_mask, attn_bias=attn_bias)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        if encoder_out is not None:
            residual = x
            if not self.post_ln:
                x = self.encoder_attn_layer_norm(x)
            x = self.encoder_attn(query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask, attn_bias=encoder_attn_bias)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if self.post_ln:
                x = self.encoder_attn_layer_norm(x)
        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        return x


def fill_with_neg_inf(t):
    return t.fill_(float('-inf'))


def bulid_future_mask(seq_len):
    return torch.triu(fill_with_neg_inf(torch.zeros([seq_len, seq_len])), 1)


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    sign = torch.sign(relative_position)
    num_buckets //= 2
    n = torch.abs(relative_position)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    max_bucket_val = num_buckets - 1 - max_exact
    val_if_large = max_exact + torch.ceil(torch.log(n.float() / max_exact) / math.log((max_distance - 1) / max_exact) * max_bucket_val).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret = torch.where(is_small, n, val_if_large) * sign
    return ret


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layers: 'int'=6, embed_dim: 'int'=768, ffn_embed_dim: 'int'=3072, attention_heads: 'int'=8, emb_dropout: 'float'=0.1, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.0, max_seq_len: 'int'=256, activation_fn: 'str'='gelu', rel_pos: 'bool'=True, rel_pos_bins: 'int'=32, max_rel_pos: 'int'=128, post_ln: 'bool'=False, auto_regressive: 'bool'=True) ->None:
        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        self.auto_regressive = auto_regressive
        if self.auto_regressive:
            self._future_mask = bulid_future_mask(self.max_seq_len)
        else:
            self._future_mask = None
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None
        self.layers = nn.ModuleList([TransformerDecoderLayer(embed_dim=self.embed_dim, ffn_embed_dim=ffn_embed_dim, attention_heads=attention_heads, dropout=dropout, attention_dropout=attention_dropout, activation_dropout=activation_dropout, activation_fn=activation_fn, post_ln=post_ln) for _ in range(decoder_layers)])
        self.rel_pos = rel_pos
        if self.rel_pos:
            assert rel_pos_bins % 2 == 0
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.relative_attention_bias = nn.Embedding(self.rel_pos_bins, self.attention_heads)
            seq_len = self.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = relative_position_bucket(relative_position, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
            self.rp_bucket -= self.rp_bucket.min()

    def get_rel_pos_bias(self, x):
        if self.rp_bucket.device != x.device:
            self.rp_bucket = self.rp_bucket
        seq_len = x.size(1)
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_future_mask(self, x, attn_mask):
        if not self.auto_regressive:
            return attn_mask
        if self._future_mask.device != x.device:
            self._future_mask = self._future_mask
        if self._future_mask.dtype != x.dtype:
            self._future_mask = self._future_mask.type_as(x)
        if attn_mask is None:
            ret = self._future_mask[:x.size(1), :x.size(1)]
            ret = ret.contiguous().unsqueeze(0).repeat(x.size(0) * self.attention_heads, 1, 1)
            return ret
        else:
            assert list(attn_mask.size()) == [x.size(0) * self.attention_heads, x.size(1), x.size(1)]
            return attn_mask + self._future_mask[:x.size(1), :x.size(1)]

    def forward(self, emb, encoder_out: 'Optional[torch.Tensor]'=None, padding_mask: 'Optional[torch.Tensor]'=None, encoder_padding_mask: 'Optional[torch.Tensor]'=None, attn_mask: 'Optional[torch.Tensor]'=None, encoder_attn_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        rel_pos_bias = self.get_rel_pos_bias(x).repeat(x.size(0), 1, 1) if self.rel_pos else None
        if attn_mask is None:
            attn_mask = rel_pos_bias
        elif rel_pos_bias is not None:
            attn_mask += rel_pos_bias
        if self.auto_regressive:
            attn_mask = self.get_future_mask(x, attn_mask)
        if attn_mask is not None and padding_mask is not None:
            attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
            attn_mask.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None
        for layer in self.layers:
            x = layer(x, encoder_out=encoder_out, padding_mask=padding_mask, attn_bias=attn_mask, encoder_padding_mask=encoder_padding_mask, encoder_attn_bias=encoder_attn_mask)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, embed_dim: 'int'=768, ffn_embed_dim: 'int'=3072, attention_heads: 'int'=8, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.0, activation_fn: 'str'='gelu', post_ln=False) ->None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = SelfMultiheadAttention(self.embed_dim, attention_heads, dropout=attention_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln

    def forward(self, x: 'torch.Tensor', attn_bias: 'Optional[torch.Tensor]'=None, padding_mask: 'Optional[torch.Tensor]'=None, return_attn: 'bool'=False) ->torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(query=x, key_padding_mask=padding_mask, attn_bias=attn_bias, return_attn=return_attn)
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layers: 'int'=6, embed_dim: 'int'=768, ffn_embed_dim: 'int'=3072, attention_heads: 'int'=8, emb_dropout: 'float'=0.1, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.0, max_seq_len: 'int'=256, activation_fn: 'str'='gelu', rel_pos: 'bool'=True, rel_pos_bins: 'int'=32, max_rel_pos: 'int'=128, post_ln: 'bool'=False) ->None:
        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim=self.embed_dim, ffn_embed_dim=ffn_embed_dim, attention_heads=attention_heads, dropout=dropout, attention_dropout=attention_dropout, activation_dropout=activation_dropout, activation_fn=activation_fn, post_ln=post_ln) for _ in range(encoder_layers)])
        self.rel_pos = rel_pos
        if self.rel_pos:
            assert rel_pos_bins % 2 == 0
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.relative_attention_bias = nn.Embedding(self.rel_pos_bins, self.attention_heads)
            seq_len = self.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = relative_position_bucket(relative_position, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
            self.rp_bucket -= self.rp_bucket.min()

    def get_rel_pos_bias(self, x):
        if self.rp_bucket.device != x.device:
            self.rp_bucket = self.rp_bucket
        seq_len = x.size(1)
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def forward(self, emb: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None, padding_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        rel_pos_bias = self.get_rel_pos_bias(x).repeat(x.size(0), 1, 1) if self.rel_pos else None
        if attn_mask is None:
            attn_mask = rel_pos_bias
        elif rel_pos_bias is not None:
            attn_mask += rel_pos_bias
        if attn_mask is not None and padding_mask is not None:
            attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
            attn_mask.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, attn_bias=attn_mask)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CrossMultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SelfMultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_dptech_corp_Uni_Core(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

