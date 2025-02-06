import sys
_module = sys.modules[__name__]
del sys
benchmarks = _module
datasets = _module
mnist = _module
wikitext2_data = _module
benchmark_dataset = _module
benchmark_mevo = _module
experimental_async_approaches = _module
offload = _module
sync_batchnorm = _module
fsdp = _module
golden_configs = _module
lm_wikitext2 = _module
oss_mnist = _module
models = _module
transformer_lm = _module
moe = _module
oss = _module
pipe = _module
utils = _module
conf = _module
fairscale = _module
experimental = _module
nn = _module
ampnet_pipe = _module
ampnet = _module
pipe = _module
auto_shard = _module
data_parallel = _module
gossip = _module
distributed = _module
gossiper = _module
graph_manager = _module
mixing_manager = _module
cuda_metering = _module
helpers = _module
distributed_pipeline = _module
data = _module
graph = _module
loss = _module
partition_handler = _module
pipeline = _module
trace = _module
mevo = _module
offload = _module
sync_batchnorm = _module
optim = _module
dynamic_loss_scaler = _module
tooling = _module
layer_memory_tracker = _module
wgit = _module
cli = _module
pygit = _module
repo = _module
sha1_store = _module
signal_sparsity = _module
signal_sparsity_profiling = _module
version = _module
fair_dev = _module
common_paths = _module
testing = _module
golden_testing_data = _module
testing = _module
testing_memory = _module
internal = _module
containers = _module
object = _module
parallel = _module
params = _module
reduce_scatter_bucketer = _module
state_dict = _module
version = _module
nn = _module
checkpoint = _module
checkpoint_activations = _module
checkpoint_utils = _module
data_parallel = _module
fsdp_optim_utils = _module
fully_sharded_data_parallel = _module
sharded_ddp = _module
misc = _module
flatten_params_wrapper = _module
param_bucket = _module
model_parallel = _module
cross_entropy = _module
initialize = _module
layers = _module
mappings = _module
random = _module
utils = _module
moe_layer = _module
top2gate = _module
async_pipe = _module
async_pipeline = _module
async_schedule = _module
balance = _module
blockpartition = _module
profile = _module
batchnorm = _module
checkpoint = _module
copy = _module
dependency = _module
messages = _module
microbatch = _module
phony = _module
pipe = _module
pipeline = _module
rpc = _module
skip = _module
layout = _module
namespace = _module
portal = _module
skippable = _module
tracker = _module
stream = _module
types = _module
worker = _module
wrap = _module
auto_wrap = _module
optim = _module
adam = _module
adascale = _module
grad_scaler = _module
layerwise_gradient_scaler = _module
oss = _module
release_utils = _module
setup = _module
tests = _module
ampnet_pipe_process = _module
test_ampnet_pipe = _module
test_gossip = _module
test_auto_shard = _module
test_mevo = _module
test_multiprocess_pipe = _module
test_offload = _module
test_sync_batchnorm = _module
test_dynamic_loss_scaler = _module
test_layer_memory_tracker = _module
test_api = _module
test_cli = _module
test_pygit = _module
test_sha1_store = _module
test_signal_sparsity = _module
test_signal_sparsity_profiling = _module
test_checkpoint_activations = _module
test_checkpoint_activations_norm = _module
test_fsdp = _module
test_fsdp_apply = _module
test_fsdp_freezing_weights = _module
test_fsdp_fwd_fwd_bwd_bwd = _module
test_fsdp_grad_acc = _module
test_fsdp_hf_transformer_eval = _module
test_fsdp_input = _module
test_fsdp_memory = _module
test_fsdp_metadata = _module
test_fsdp_multiple_forward = _module
test_fsdp_multiple_forward_checkpoint = _module
test_fsdp_multiple_wrapping = _module
test_fsdp_optimizer_utils = _module
test_fsdp_overlap = _module
test_fsdp_pre_backward_hook = _module
test_fsdp_regnet = _module
test_fsdp_shared_weights = _module
test_fsdp_shared_weights_mevo = _module
test_fsdp_state_dict = _module
test_fsdp_summon_full_params = _module
test_fsdp_uneven = _module
test_fsdp_with_checkpoint_wrapper = _module
test_sharded_ddp_features = _module
test_sharded_ddp_pytorch_parity = _module
test_flatten_params_wrapper = _module
test_grad_bucket = _module
test_param_bucket = _module
test_cross_entropy = _module
test_initialize = _module
test_layers = _module
test_random = _module
test_moe_layer = _module
test_top2gating = _module
conftest = _module
test_api = _module
test_gpipe = _module
test_inspect_skip_layout = _module
test_leak = _module
test_portal = _module
test_stash_pop = _module
test_tracker = _module
test_verify_skippables = _module
test_balance = _module
test_bugs = _module
test_checkpoint = _module
test_checkpoint_ddp = _module
test_copy = _module
test_deferred_batch_norm = _module
test_dependency = _module
test_inplace = _module
test_microbatch = _module
test_parity = _module
test_phony = _module
test_pipe = _module
test_pipeline = _module
test_stream = _module
test_transparency = _module
test_worker = _module
pipe_process = _module
conftest = _module
test_bugs = _module
test_inplace = _module
test_pipe = _module
test_rpc = _module
test_transparency = _module
test_wrap = _module
test_adam = _module
test_ddp_adascale = _module
test_layerwise_gradient_scaler = _module
test_oss = _module
test_oss_adascale = _module
test_single_node_adascale = _module
test_containers = _module
test_parallel = _module
test_reduce_scatter_bucketer = _module
test_state_dict = _module
test_version = _module

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


from collections import namedtuple


import torch


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import Dataset


import time


from torch import nn


from torch.cuda import Event


import logging


import math


import warnings


from torch.distributed import rpc


import torch.multiprocessing as mp


import torch.nn as nn


from torch.optim.optimizer import Optimizer


from functools import reduce


import numpy as np


from torch.optim import Adam


from torch.utils.data.dataloader import DataLoader


from torchvision.datasets import FakeData


from torchvision.transforms import ToTensor


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from collections import defaultdict


from enum import Enum


from typing import Any


from typing import List


from typing import Optional


from typing import cast


import torch.autograd.profiler as profiler


from torch.cuda.amp import GradScaler as TorchGradScaler


from torch.utils.data import BatchSampler


from torch.utils.data import Sampler


from torchvision.datasets import MNIST


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from typing import Dict


from typing import Tuple


from typing import Union


from torch.autograd.profiler import record_function


from torch.distributed import ProcessGroup


from typing import Set


import torch.fx


from torch.fx.node import Node


import functools


from typing import Callable


from typing import Iterable


from torch.autograd import Variable


from torch.nn.modules import Module


from typing import Iterator


from abc import ABC


from abc import abstractmethod


from math import log as mlog


from collections import deque


from functools import partial


from typing import ClassVar


from typing import Deque


import collections


from typing import MutableMapping


from torch import Tensor


from torch.distributed.nn import RemoteModule


from types import TracebackType


from typing import Type


import inspect


import torch.nn.functional as F


from enum import auto


from functools import lru_cache


from typing import NamedTuple


from typing import Sequence


from torch.utils.hooks import RemovableHandle


import copy


import random


from typing import TYPE_CHECKING


from typing import Generator


import numpy


from collections import OrderedDict


from torch.nn.utils.rnn import PackedSequence


import collections.abc as abc


from math import inf


import re


import torch.utils.checkpoint as torch_checkpoint


from torch.nn.modules.batchnorm import _BatchNorm


from itertools import groupby


import typing


from typing import Mapping


from torch.nn.parameter import Parameter


from itertools import chain


import torch.nn.init as init


from torch.cuda import _lazy_call


from torch.utils.checkpoint import detach_variable


from torch.nn import Module


from torch.nn import ModuleList


import itertools


from typing import TypeVar


from torch import ByteTensor


import torch.autograd


from queue import Empty as QueueEmpty


from queue import Queue


import torch.cuda.comm


import torch.cuda


from torch.distributed.distributed_c10d import _get_global_rank


from typing import FrozenSet


from torch.optim import SGD


from torch.optim import Optimizer


from torch.cuda import FloatTensor


from torch.cuda.amp.common import amp_definitely_not_available


from torch.cuda.amp.grad_scaler import GradScaler as TorchGradScaler


from torch.optim.sgd import SGD


from torch.autograd import profiler


from torch.nn import Parameter


import torch.distributed


import torch.nn


import torch.distributed.autograd as dist_autograd


from torch.distributed.optim import DistributedOptimizer


import torch.distributed.rpc as rpc


import torch.optim as optim


from torch.nn.parallel import DistributedDataParallel


from torch.utils.checkpoint import checkpoint as torch_checkpoint_wrapper


from torch.nn import BatchNorm2d


from torch.nn import LayerNorm


from torch.nn import Linear


from torch.nn import Sequential


from itertools import product


from time import time


from torch.optim import Adadelta


from torch.cuda.amp import GradScaler


from torch.nn import AdaptiveAvgPool2d


from torch.nn import Conv2d


from torch.nn import CrossEntropyLoss


from torch.nn import ReLU


from torch.nn import Sigmoid


from torch.nn import SyncBatchNorm


from copy import deepcopy


from torch.utils.checkpoint import checkpoint as torch_checkpoint


from torch import optim


from sklearn.datasets import make_blobs


from torch.cuda.amp.autocast_mode import autocast


import torchvision


import torchvision.transforms as transforms


from torch.optim.lr_scheduler import LambdaLR


class EmbeddingLayer(nn.Embedding):
    """Wrapped nn.Embedding layer to allow for weight initialization."""

    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp_sqrt = math.sqrt(ninp)
        self.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return super().forward(src) * self.ninp_sqrt


class PositionalEncodingLayer(nn.Module):
    """PositionalEncoding layer for a given Transformer model."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeedForwardLayer(nn.Module):
    """FeedForward layer for a given Transformer model."""

    def __init__(self, d_model, dim_feedforward, activation, dropout) ->None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))


class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Any', group: 'dist.ProcessGroup', input: 'Tensor') ->Tensor:
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: 'Any', *grad_output: Tensor) ->Tuple[None, Tensor]:
        return None, _AllToAll.apply(ctx.group, *grad_output)


def gumbel_rsample(shape: 'Tuple', device: 'torch.device') ->Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(tensor: 'torch.Tensor', num_classes: 'int') ->Tensor:
    """Workaround for https://github.com/pytorch/pytorch/issues/55579"""
    assert num_classes > 0, 'num_classes must be a positive integer'
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret


def top2gating(logits: 'torch.Tensor') ->Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=1, dtype=torch.float)
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = 2 * num_tokens // num_experts
    assert num_tokens % num_experts == 0
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts)
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float('-inf'))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = one_hot(indices2_s, num_classes=num_experts)
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce)
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)
    denom_s = gates1_s + gates2_s
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    gates1 = gates1_s.unsqueeze(-1) * mask1
    gates2 = gates2_s.unsqueeze(-1) * mask2
    locations1_sc = one_hot(locations1_s, num_classes=capacity)
    locations2_sc = one_hot(locations2_s, num_classes=capacity)
    combine1_sec = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)
    combine2_sec = gates2.unsqueeze(2) * locations2_sc.unsqueeze(1)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    return l_aux, combine_weights, dispatch_mask


class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """
    wg: 'torch.nn.Linear'

    def __init__(self, model_dim: 'int', num_experts: 'int') ->None:
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, input: 'torch.Tensor') ->Tuple[Tensor, Tensor, Tensor]:
        logits = self.wg(input)
        return top2gating(logits)


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
        is_moe: if ``True``, the feedforward layer will have MOE enabled.
        num_local_experts: number of local experts for MOE.


    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU(), layer_norm_eps=1e-05, norm_first=False, is_moe=False, num_local_experts=1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.is_moe = is_moe
        if is_moe:
            world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
            num_global_experts = num_local_experts * world_size
            self.gate = Top2Gate(d_model, num_global_experts)
            experts = nn.ModuleList([FeedForwardLayer(d_model, dim_feedforward, activation, dropout) for _ in range(num_local_experts)])
            self.moe_layer = MOELayer(self.gate, experts)
        else:
            self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def _ff_block(self, x):
        if self.is_moe:
            return self.moe_layer(x)
        else:
            return self.ff_block(x)


class TransformerDecoderLayer(TransformerEncoderLayer):
    """TransformerDecoder layer which inherits from TransformerEncoderLayer."""

    def __init__(self, ninp, nhead, nhid, dropout, is_moe=False, num_local_experts=1):
        super().__init__(ninp, nhead, nhid, dropout, is_moe=is_moe, num_local_experts=num_local_experts)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask
        return super().forward(src, self.src_mask)


class LinearLayer(nn.Linear):
    """Wrapped nn.Linear layer to allow for weight initialization."""

    def __init__(self, ninp, ntoken, initrange):
        super().__init__(ninp, ntoken)
        self.bias.data.zero_()
        self.weight.data.uniform_(-initrange, initrange)


class TransformerLMSequntial(nn.Sequential):
    """A small language model based on the design of GPT-2 using nn.Sequeitnal
    for compatability with Pipe"""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder):
        layers = [EmbeddingLayer(ntokens, ninp, initrange), PositionalEncodingLayer(ninp, dropout)]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout))
        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLMSequntial, self).__init__(*layers)


class TransformerLM(nn.Sequential):
    """A GPT-2 based nn.Sequential language model."""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder, is_moe=False, num_local_experts=1):
        layers = [EmbeddingLayer(ntokens, ninp, initrange), PositionalEncodingLayer(ninp, dropout)]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout, is_moe, num_local_experts))
        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLM, self).__init__(*layers)


BROADCAST_BUCKET_SIZE = 10 * 1024 * 1024


HEARTBEAT_TIMEOUT = 300


class MixingManager(ABC):

    def __init__(self, graph: 'GraphManager', device: 'Optional[torch.device]') ->None:
        self.graph_manager = graph
        self.device = device

    def is_regular(self) ->bool:
        """
        Whether there is bias accumulated in local entry of stationary
        distribution of mixing matrix
        """
        return self.graph_manager.is_regular_graph() and self.is_uniform()

    @abstractmethod
    def is_uniform(self) ->bool:
        """Whether mixing weights are distributed uniformly over peers"""
        raise NotImplementedError

    @abstractmethod
    def get_mixing_weights(self, residual_adjusted: 'bool'=True) ->Dict[Union[str, int], torch.Tensor]:
        """Create mixing weight dictionary using uniform allocation"""
        raise NotImplementedError


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    Creates an adapter to make logging for multiple processes cleaner
    """

    def process(self, msg: 'str', kwargs: 'Any') ->Tuple[str, MutableMapping[str, Any]]:
        process_num = kwargs.pop('process_num', self.extra['process_num'])
        return f'process: {process_num} {msg}', kwargs


class UniformMixing(MixingManager):

    def get_mixing_weights(self, residual_adjusted: 'bool'=True) ->Dict[Union[str, int], torch.Tensor]:
        """Create mixing weight dictionary using uniform allocation"""
        mixing_weights: 'Dict[Union[str, int], torch.Tensor]' = {}
        out_peers, _ = self.graph_manager.get_peers()
        w = torch.tensor([1.0 / (len(out_peers) + 1.0)], device=self.device)
        mixing_weights['lo'] = w.clone()
        w_op = w if not residual_adjusted else w / mixing_weights['lo']
        mixing_weights['uniform'] = w_op.clone()
        for op in out_peers:
            mixing_weights[op] = w_op.clone()
        return mixing_weights

    def is_uniform(self) ->bool:
        return True


class dist_backend(str, Enum):
    UNDEFINED = 'undefined'
    TCP = 'tcp'
    MPI = 'mpi'
    GLOO = 'gloo'
    NCCL = 'nccl'


class SlowMoBaseAlgorithm(str, Enum):
    LOCALSGD = 'localsgd'
    SGP = 'sgp'


def flatten_tensors(tensors: 'List[torch.Tensor]') ->torch.Tensor:
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually
    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten
    Returns:
        A 1D buffer containing input tensors
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def group_by_dtype(tensors: 'List[torch.Tensor]') ->Dict[torch.dtype, List[torch.Tensor]]:
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arg:
        tensors (Iterable[Tensor]): list of tensors
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def unflatten_tensors(flat: 'torch.Tensor', tensors: 'List[torch.Tensor]') ->List[torch.Tensor]:
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Args:
        flat (Tensor): flattened dense tensors to unflatten
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs


def communicate(tensors: 'List[torch.Tensor]', communication_op: 'Any', logger: 'logging.Logger'=None) ->None:
    """
    Communicate a list of tensors
    Args:
        tensors (Iterable[Tensor]): list of tensors
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce
    """
    tensors_by_dtype = group_by_dtype(tensors)
    for tensors_with_same_dtype in tensors_by_dtype.values():
        flat_tensor = flatten_tensors(tensors_with_same_dtype)
        if logger is not None:
            logger.debug('Flatten completed')
        communication_op(tensor=flat_tensor)
        if logger is not None:
            logger.debug('Commmunication completed')
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(flat_tensor, tensors_with_same_dtype), tensors_with_same_dtype):
                t.copy_(f)
        if logger is not None:
            logger.debug('Unflatten completed')


def create_and_record_event() ->torch.Event:
    event = torch.Event(enable_timing=True)
    event.record()
    return event


MAX_LEN_DEQUEUE = 10 ** 4


deque_with_max_len_fixed = partial(deque, maxlen=MAX_LEN_DEQUEUE)


def create_process_group(ranks: 'List[int]') ->torch.distributed.ProcessGroup:
    """
    Creates and intializes a new process group. Assumes init_process_group
    has already been called
    Arguments:
        ranks (list<int>): ranks corresponding to the processes which should
            belong the created process group
    Returns:
        New process group
    """
    new_group = dist.new_group(ranks=ranks)
    init_tensor_fp32, init_tensor_fp16 = torch.zeros(1), torch.zeros(1).half()
    for init_tensor in [init_tensor_fp32, init_tensor_fp16]:
        if torch.cuda.is_available():
            init_tensor = init_tensor
        if dist.get_rank() in ranks:
            dist.all_reduce(init_tensor, group=new_group)
        torch.cuda.synchronize()
    return new_group


def make_logger(rank: 'int', verbose: 'bool'=True) ->logging.Logger:
    """
    Return a logger for writing to stdout
    Args:
        rank (int): rank of node making logger
        verbose (bool): whether to set log-level to INFO; o.w. WARNING
    Returns:
        Python logger
    """
    logger = logging.getLogger(__name__)
    if logger not in HANDLER_AND_LEVEL_SET:
        console = logging.StreamHandler(stream=sys.stdout)
        format_str = '{}'.format(rank)
        format_str += ': %(levelname)s -- %(threadName)s -- %(message)s'
        console.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console)
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        HANDLER_AND_LEVEL_SET.add(logger)
    return logger


class MultiInputSequential(nn.Module):
    """A variation of nn.Sequential, that allows the first module in the sequence accepts
    multiple inputs. To be used internally by _split_module
    """

    def __init__(self, *modules: nn.Module) ->None:
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, *inputs: Tuple[Tensor]) ->Tensor:
        input = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            input = module(input)
        return input


ConsumerType = TypeVar('ConsumerType')


def RemoteSequential(rref_list: 'List[rpc.RRef]') ->MultiInputSequential:
    return MultiInputSequential(*(r.local_value() for r in rref_list))


Tensors = Tuple[Tensor, ...]


TensorOrTensors = Union[Tensor, Tensors]


class Batch:
    """An abstraction of an atomic tensor or a tuple of tensors. This
    eliminates every boilerplate code to classify an atomic tensor or a tuple
    of tensors.
    ::

        x = generate_tensor_or_tensors()
        x = Batch(x)

        # in-place update
        x[0] = F.apply(x[0])
        x[:] = F.apply(*x)

        # f(x) if x is a tensor.
        # f(*x) if x is a tuple of tensors.
        # y is also a batch.
        y = x.call(f)

    """

    def __init__(self, value: 'TensorOrTensors', index: 'int') ->None:
        self.value = value
        self.atomic = torch.is_tensor(value)
        self.__index = index

    @property
    def index(self) ->int:
        return self.__index

    @property
    def tensor(self) ->Tensor:
        """Retrieves the underlying tensor."""
        if not self.atomic:
            raise AttributeError('not atomic batch')
        return cast(Tensor, self.value)

    @property
    def tensors(self) ->Tensors:
        """Retrieves the underlying tensors."""
        if self.atomic:
            raise AttributeError('batch is atomic')
        return cast(Tensors, self.value)

    @property
    def tensor_or_tensors(self) ->TensorOrTensors:
        """Retrieves the underlying tensor or tensors regardless of type."""
        return self.value

    def call(self, function: 'Function') ->'Batch':
        """Calls a function by the underlying tensor or tensors. It also wraps
        the output with :class:`Batch`.
        """
        return Batch(function(self.value), self.index)

    def __repr__(self) ->str:
        return f'Batch[atomic={self.atomic!r}]({self.value!r})'

    def __iter__(self) ->Iterator[Tensor]:
        if self.atomic:
            yield self.tensor
        else:
            yield from self.tensors

    def __len__(self) ->int:
        return 1 if self.atomic else len(self.tensors)

    def __getitem__(self, index: 'int') ->Tensor:
        if not self.atomic:
            return self.tensors[index]
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        return self.tensor

    @typing.overload
    def __setitem__(self, index: 'int', value: 'Tensor') ->None:
        ...

    @typing.overload
    def __setitem__(self, index: 'slice', value: 'Tensors') ->None:
        ...

    def __setitem__(self, index: 'Union[int, slice]', value: 'TensorOrTensors') ->None:
        if isinstance(index, int):
            value = cast(Tensor, value)
            self._setitem_by_index(index, value)
        else:
            value = cast(Tensors, value)
            self._setitem_by_slice(index, value)

    def _setitem_by_index(self, index: 'int', value: 'Tensor') ->None:
        if not self.atomic:
            i = index
            self.value = self.value[:i] + (value,) + self.value[i + 1:]
            return
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        self.value = value

    def _setitem_by_slice(self, index: 'slice', value: 'Tensors') ->None:
        if not index.start is index.stop is index.step is None:
            raise NotImplementedError('only slice [:] supported')
        if not self.atomic:
            self.value = value
            return
        if len(value) != 1:
            raise IndexError('atomic batch cannot be replaced with multiple tensors')
        self.value = value[0]


class CPUStreamType:
    pass


AbstractStream = Union[torch.Stream, CPUStreamType]


CPUStream = CPUStreamType()


def default_stream(device: 'torch.device') ->AbstractStream:
    """:func:`torch.cuda.default_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.default_stream(device)


def as_cuda(stream: 'AbstractStream') ->torch.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.Stream, stream)


def is_cuda(stream: 'Optional[AbstractStream]') ->bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream


def get_phony(device: 'torch.device', *, requires_grad: bool) ->Tensor:
    """Gets a phony. Phony is tensor without space. It is useful to make
    arbitrary dependency in a autograd graph because it doesn't require any
    gradient accumulation.

    .. note::

        Phonies for each device are cached. If an autograd function gets a phony
        internally, the phony must be detached to be returned. Otherwise, the
        autograd engine will mutate the cached phony in-place::

            class Phonify(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    phony = get_phony(input.device, requires_grad=False)
                    return phony.detach()  # detach() is necessary.

    """
    key = device, requires_grad
    try:
        phony = _phonies[key]
    except KeyError:
        with use_stream(default_stream(device)):
            phony = torch.empty(1, device=device, requires_grad=requires_grad)
        _phonies[key] = phony
    return phony


class Fork(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "'Fork'", input: 'Tensor') ->Tuple[Tensor, Tensor]:
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: "'Fork'", grad_input: 'Tensor', grad_grad: 'Tensor') ->Tensor:
        return grad_input


def fork(input: 'Tensor') ->Tuple[Tensor, Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)
    return input, phony


class Join(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "'Join'", input: 'Tensor', phony: 'Tensor') ->Tensor:
        return input.detach()

    @staticmethod
    def backward(ctx: "'Join'", grad_input: 'Tensor') ->Tuple[Tensor, None]:
        return grad_input, None


def join(input: 'Tensor', phony: 'Tensor') ->Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)
    return input


MOVING_DENIED = TypeError('denied to move parameters and buffers, because Pipe should manage device placement')


def save_rng_states(device: 'torch.device', rng_states: 'Deque[RNGStates]') ->None:
    """:meth:`Checkpoint.forward` captures the current PyTorch's random number
    generator states at CPU and GPU to reuse in :meth:`Recompute.backward`.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state: 'Optional[ByteTensor]'
    if device.type == 'cuda':
        gpu_rng_state = torch.get_rng_state(device)
    else:
        gpu_rng_state = None
    rng_states.clear()
    rng_states.append((cpu_rng_state, gpu_rng_state))


class Checkpoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Context', phony: 'Tensor', recomputed: 'Deque[Recomputed]', rng_states: 'Deque[RNGStates]', function: 'Function', input_atomic: 'bool', *input: Tensor) ->TensorOrTensors:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states
        save_rng_states(input[0].device, ctx.rng_states)
        ctx.function = function
        ctx.input_atomic = input_atomic
        ctx.save_for_backward(*input)
        with torch.no_grad(), enable_checkpointing():
            output = function(input[0] if input_atomic else input)
        return output

    @staticmethod
    def backward(ctx: 'Context', *grad_output: Tensor) ->Tuple[Optional[Tensor], ...]:
        output, input_leaf = ctx.recomputed.pop()
        if isinstance(output, tuple):
            tensors = output
        else:
            tensors = output,
        if any(y.requires_grad for y in tensors):
            tensors = tuple([x for x in tensors if x.requires_grad])
            torch.autograd.backward(tensors, grad_output)
        grad_input: 'List[Optional[Tensor]]' = [None, None, None, None, None]
        grad_input.extend(x.grad for x in input_leaf)
        return tuple(grad_input)


class Recompute(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Context', phony: 'Tensor', recomputed: 'Deque[Recomputed]', rng_states: 'Deque[RNGStates]', function: 'Function', input_atomic: 'bool', *input: Tensor) ->Tensor:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states
        ctx.function = function
        ctx.input_atomic = input_atomic
        ctx.save_for_backward(*input)
        return phony

    @staticmethod
    def backward(ctx: 'Context', *grad_output: Tensor) ->Tuple[None, ...]:
        input = ctx.saved_tensors
        input_leaf = tuple(x.detach().requires_grad_(x.requires_grad) for x in input)
        with restore_rng_states(input[0].device, ctx.rng_states):
            with torch.enable_grad(), enable_recomputing():
                output = ctx.function(input_leaf[0] if ctx.input_atomic else input_leaf)
        ctx.recomputed.append((output, input_leaf))
        grad_input: 'List[None]' = [None, None, None, None, None]
        grad_input.extend(None for _ in ctx.saved_tensors)
        return tuple(grad_input)


class Checkpointing:
    """Generates a pair of :class:`Checkpoint` and :class:`Recompute`."""

    def __init__(self, function: 'Function', batch: 'Batch') ->None:
        self.function = function
        self.batch = batch
        self.recomputed: 'Deque[Recomputed]' = deque(maxlen=1)
        self.rng_states: 'Deque[RNGStates]' = deque(maxlen=1)

    def checkpoint(self) ->Batch:
        """Returns a batch applied by :class:`Checkpoint`."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)
        phony = get_phony(self.batch[0].device, requires_grad=True)
        output = Checkpoint.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *input)
        if isinstance(output, tuple):
            output = tuple([(x if x.is_floating_point() else x.detach()) for x in output])
        return Batch(output, self.batch.index)

    def recompute(self, batch: 'Batch') ->None:
        """Applies :class:`Recompute` to the batch in place."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)
        batch[0], phony = fork(batch[0])
        phony = Recompute.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *input)
        batch[0] = join(batch[0], phony)


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


class Task:
    """A task represents how to compute a micro-batch on a partition.

    It consists of two parts: :meth:`compute` and :meth:`finalize`.
    :meth:`compute` should be executed in worker threads concurrently.
    :meth:`finalize` should be executed after when worker threads complete to
    execute :meth:`compute`.

    :meth:`compute` might be boosted by worker threads. Because it produces
    several CUDA API calls by user code. In PyTorch, parallel CUDA API calls
    are not serialized through GIL. So more than one CUDA API call can be
    produced at the same time.

    """

    def __init__(self, stream: 'Optional[AbstractStream]', *, compute: Callable[[], Batch], finalize: Optional[Callable[[Batch], None]]) ->None:
        self.stream = stream
        self._compute = compute
        self._finalize = finalize
        self._grad_enabled = torch.is_grad_enabled()

    def compute(self) ->Batch:
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):
            return self._compute()

    def finalize(self, batch: 'Batch') ->None:
        if self._finalize is None:
            return
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):
            self._finalize(batch)


def worker(in_queue: 'InQueue', out_queue: 'OutQueue', device: 'torch.device') ->None:
    """The main loop of a worker thread."""
    with use_device(device):
        while True:
            task = in_queue.get()
            if task is None:
                break
            try:
                batch = task.compute()
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                out_queue.put((False, exc_info))
                continue
            out_queue.put((True, (task, batch)))
    done = False, None
    out_queue.put(done)


def current_stream(device: 'torch.device') ->AbstractStream:
    """:func:`torch.cuda.current_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.current_stream(device)


def torch_version(version: 'str'=torch.__version__) ->Tuple[int, ...]:
    numbering = re.search('^(\\d+).(\\d+).(\\d+)([^\\+]*)(\\+\\S*)?$', version)
    if not numbering:
        return tuple()
    global _logged
    if numbering.group(4) and not _logged:
        logging.warning(f'Pytorch pre-release version {version} - assuming intent to test it')
        _logged = True
    return tuple(int(numbering.group(n)) for n in range(1, 4))


def check_pytorch_version() ->None:
    if torch_version() < (1, 8, 0):
        raise Exception('DistributedPipeline requires PyTorch version 1.8 or higher')


def _reshape_inputs(input: 'torch.Tensor', target: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """Convert 3D inputs to 2D for this kernel"""
    if len(input.shape) == 3:
        input = input.reshape(-1, input.shape[2])
    if len(target.shape) == 2:
        target = target.reshape(-1)
    return input, target


class BaselineSoftmax(nn.Module):
    """Baseline softmax that does an output linear projection and a softmax.


        We also support LMCL (Large Margin Cosine Loss) from the CosFace paper. See
        more detailed comment in the MEVO class below.

        This is intended to be used with an embedding layer with shared weights.

    Args:
        proj_weight (nn.Parameter):
            The shared weight.
        tile_factor (int):
            Unused. It is here to make kernel init easier with MEVO.
        log_softmax (bool):
            If True, use log_softmax instead of softmax.
        margin (float):
            Used in LMCL (when scale != None). See MEVO comments for
            more details.
        scale (Optional[float]):
            Used in LMCL. If scale is None, LMCL is turned off. See
            MEVO comments for more details.

    """

    def __init__(self, proj_weight: 'nn.Parameter', tile_factor: 'int'=0, log_softmax: 'bool'=True, margin: 'float'=0.35, scale: 'Optional[float]'=None):
        super().__init__()
        out_dim, in_dim = proj_weight.shape
        assert 'cuda' in str(proj_weight.device), 'weight should be on GPU'
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        assert proj_weight.dtype in [torch.float16, torch.float32]
        if proj_weight.dtype == torch.float16:
            self.fc = self.fc.half()
        self.fc.weight = proj_weight
        assert self.fc.weight.dtype in [torch.float16, torch.float32], self.fc.weight.dtype
        self.fp16 = self.fc.weight.dtype == torch.float16
        self.log_softmax = log_softmax
        self.margin = margin
        self.scale = scale

    def lmcl_pre_softmax(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        x = F.normalize(input, dim=1)
        w = F.normalize(self.fc.weight, dim=1)
        logits = torch.einsum('nc,kc->nk', x, w)
        row_ind = torch.arange(x.shape[0], dtype=torch.long)
        col_ind = target
        logits[row_ind, col_ind] -= self.margin
        logits *= self.scale
        return logits

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        """Forward function that computes softmax output with the input and target."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        if self.fp16:
            assert input.dtype == torch.float16
        if self.scale is not None:
            x = self.lmcl_pre_softmax(input, target)
        else:
            x = self.fc(input)
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1, dtype=torch.float32)
        else:
            x = F.softmax(x, dim=-1, dtype=torch.float32)
        assert x.dtype == torch.float32
        return x


class BaselineSoftmaxNllLoss(BaselineSoftmax):
    """Baseline that does an output projection, a softmax & a NLL loss (cross-entropy).

    See BaselineSoftmax above. Constructor is the same. Only difference is in the
    forward function.

    This class is used for testing and benchmarking.
    """

    def __init__(self, proj_weight: 'nn.Parameter', tile_factor: 'int'=0, log_softmax: 'bool'=True, margin: 'float'=0.35, scale: 'Optional[float]'=None):
        super().__init__(proj_weight, tile_factor, log_softmax, margin, scale)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        """Forward that directly compute the loss."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        x = super().forward(input, target)
        return F.nll_loss(x, target, reduction='sum')


DEBUG = False


class BackwardTriggerFn(torch.autograd.Function):
    """A backward trigger function."""

    @staticmethod
    def forward(ctx: 'Any', w: 'torch.Tensor', trigger_tensor: 'torch.Tensor') ->torch.Tensor:
        """We take a weight tensor and the trigger as inputs and output the weight directly."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(w, trigger_tensor)
        return w

    @staticmethod
    def backward(ctx: 'Any', *args: Any) ->Any:
        """We return zero grad for the trigger only."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        w, trigger = ctx.saved_tensors
        assert w.requires_grad
        assert trigger.requires_grad
        return None, torch.zeros_like(trigger)


class BackwardTrigger(nn.Module):
    """A backward trigger module.

    This module takes a parameter as an input and create a linked parameter
    from a newly created trigger parameter.

    The way to use it in a module's ``__init__'' and ``forward'' functions:

    ```
    def __init__():
      ...
      self.trigger = BackwardTrigger(some_layer.weight)
      ...

    def forward():
      w = self.trigger()
      ... continue to use w ...
    ```

    As a resule, the trigger's backward hook will be called at the end of
    the backward for the module that uses this trigger.
    """

    def __init__(self, linked_param: 'torch.Tensor'):
        super().__init__()
        assert isinstance(linked_param, nn.Parameter)
        self.trigger = nn.Parameter(torch.rand(1, dtype=linked_param.dtype, device=linked_param.device))
        self.trigger._linked_param = linked_param

    def forward(self) ->torch.Tensor:
        return BackwardTriggerFn.apply(self.trigger._linked_param, self.trigger)


def lmcl_matmul(i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', w_idx: 'int', margin: 'float', scale: 'Optional[float]') ->torch.Tensor:
    """LMCL variation of matmul with normalization, margin and scale."""
    logits = torch.matmul(F.normalize(i, dim=1), F.normalize(w, dim=1).T)
    mask = torch.arange(w_idx * w.shape[0], (w_idx + 1) * w.shape[0], dtype=torch.long, device=i.device).expand(i.shape[0], -1)
    logits[mask == tgt.reshape(-1, 1)] -= margin
    logits *= scale
    return logits


class GetMaxFunction(torch.autograd.Function):
    """Custom checkpointed function to get max-per-token from an input and a weight"""

    @staticmethod
    def get_max(i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', w_idx: 'int', full_precision: 'bool', margin: 'float', scale: 'Optional[float]') ->torch.Tensor:
        """
        Throughout this code:

          i: input data with shape = (split-of-tokens, d_model)
          w: weight data with shape = (split-of-vocabs, d_model)
          tgt: target prediction data with shape = (split-of-tokens,)
        """
        if scale is not None:
            _m = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _m = torch.matmul(i, w.T)
        if full_precision:
            _m = _m.float()
        _m = _m.max(dim=1)[0]
        return _m

    @staticmethod
    def forward(ctx: 'Any', i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', kernel_obj: "'MemoryEfficientVocabOutput'", w_idx: 'int', w_split_size: 'int', split_dim: 'int') ->torch.Tensor:
        """Forward function that computes the max, without saving activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, tgt)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        ctx.args = {}
        assert split_dim == 0
        with torch.no_grad():
            return GetMaxFunction.get_max(i, w, tgt, w_idx, kernel_obj.fp_max, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: 'Any', *args: Any) ->Any:
        """Recompute the forward max and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, tgt = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            maxs = GetMaxFunction.get_max(i, w, tgt, ctx.w_idx, ctx.kernel_obj.fp_max, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(maxs, *args)
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, None, None, None, None


class GetSumFunction(torch.autograd.Function):
    """Custom checkpointed function to get sum-per-token from an input and a weight."""

    @staticmethod
    def get_sum(i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', maxs: 'torch.Tensor', w_idx: 'int', full_precision: 'bool', margin: 'float', scale: 'Optional[float]') ->torch.Tensor:
        if scale is not None:
            _s = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _s = torch.matmul(i, w.T)
        if full_precision:
            _s = _s.float()
        _s = (_s - maxs.reshape(-1, 1)).exp().sum(dim=1)
        return _s

    @staticmethod
    def forward(ctx: 'Any', i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', maxs: 'torch.Tensor', kernel_obj: "'MemoryEfficientVocabOutput'", w_idx: 'int', w_split_size: 'int', split_dim: 'int') ->torch.Tensor:
        """Forward function that computes the sum, without saving activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, tgt, maxs)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        assert split_dim == 0
        with torch.no_grad():
            return GetSumFunction.get_sum(i, w, tgt, maxs, w_idx, kernel_obj.fp_sum, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: 'Any', *args: Any) ->Any:
        """Recompute the forward sum and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, tgt, maxs = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert maxs.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        maxs = maxs.detach().requires_grad_(True)
        with torch.enable_grad():
            sums = GetSumFunction.get_sum(i, w, tgt, maxs, ctx.w_idx, ctx.kernel_obj.fp_sum, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(sums, *args)
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, maxs.grad, None, None, None, None


class TargetScoreFunction(torch.autograd.Function):
    """Custom checkpointed function to compute the target score."""

    @staticmethod
    def get_target_score(i: 'torch.Tensor', w: 'torch.Tensor', target: 'torch.Tensor', full_precision: 'bool', margin: 'float', scale: 'Optional[float]') ->torch.Tensor:
        tokens, d_model = i.shape
        assert d_model == w.shape[1]
        tw = w.gather(dim=0, index=target.reshape(target.shape[0], 1).expand(target.shape[0], d_model))
        assert tw.shape == (tokens, d_model)
        if scale is not None:
            target_score = F.normalize(i, dim=1) * F.normalize(tw, dim=1)
        else:
            target_score = i * tw
        if full_precision:
            target_score = target_score.float()
        target_score = target_score.sum(dim=1)
        if scale is not None:
            target_score -= margin
            target_score *= scale
        return target_score

    @staticmethod
    def forward(ctx: 'Any', i: 'torch.Tensor', w: 'torch.Tensor', target: 'torch.Tensor', kernel_obj: "'MemoryEfficientVocabOutput'") ->torch.Tensor:
        """Forward, without activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, target)
        ctx.kernel_obj = kernel_obj
        with torch.no_grad():
            x = TargetScoreFunction.get_target_score(i, w, target, kernel_obj.fp_target, kernel_obj.margin, kernel_obj.scale)
        return x

    @staticmethod
    def backward(ctx: 'Any', *args: Any) ->Any:
        """Forward and backward again, assign or accumulate the gradients."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        i, w, target = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert not target.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            scores = TargetScoreFunction.get_target_score(i, w, target, ctx.kernel_obj.fp_target, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(scores, *args)
        if ctx.kernel_obj.proj_weight.grad is not None:
            ctx.kernel_obj.proj_weight.grad.add_(w.grad)
        else:
            ctx.kernel_obj.proj_weight.grad = w.grad
        return i.grad, None, None, None


def _next_power_of_2_or_max(n: 'int', max_n: 'int') ->int:
    """Return the smallest power of 2 greater than or equal to n, with a limit.

    Useful when used in splitting a tensor into chunks with power-of-2 sizes.
    """
    if n == 0:
        return 1
    orig_n = n
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    assert n >= orig_n, f'{n} vs. {orig_n}'
    assert bin(n).count('1') == 1, bin(n)
    if n > max_n:
        return max_n
    return n


class MemoryEfficientVocabOutput(nn.Module):
    """Fused fc + softmax + nll_loss in a tiled fashion.

        MEVO uses much less memory but is quite a bit slower.

        MEVO also implements the LMCL (Large Margin Cosine Loss) function introduced by
        highly cited
        `CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`_.

        .. _`CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`: https://arxiv.org/abs/1801.09414

        LMCL can be turned on using the ``margin`` and ``scale`` parameters below. These
        hyperparameters most likely require tuning, depending on the number of classes etc.

        MEVO LMCL can be suitable for face recognition and image retrieval tasks, esp. when
        the number prediction target classes is large. MEVO is slower but can use much
        less GPU memory in that case, which enables training with larger batches. We
        hope this is helpful but we strongly recommend users (AI researchers
        and engineers) to carefully consider their applications of this technology. This
        types of technology should not be used by small group of people exclusively to
        potentially harm the general public.

    Args:
        proj_weight (nn.Parameter):
            Sharing this weight with an embedding layer.
        tile_factor (int):
            Number of splits to use on the input sequence and vocab dimensions.
            Default: 16
        reduction (str):
            Reduction OP (sum or mean).
            Default: sum
        margin (float):
            Hyperparameter of the separation margin between classes. See the
            appendix of the CosFace paper for a formula on how to compute its
            value properly. The default value is unlikely to be suitable in all
            cases.
            Default: 0.35
        scale (Optional[float]):
            Hyperparameter of the feature-vector-scaling for LMCL. When not
            supplied, LMCL is turned off. See the appendix of the CosFace paper for
            a formula on how to compute its value properly.
            Default: None
    """

    def __init__(self, proj_weight: 'nn.Parameter', tile_factor: 'int'=16, reduction: 'str'='sum', margin: 'float'=0.35, scale: 'Optional[float]'=None):
        super().__init__()
        self.proj_weight = proj_weight
        self.tf_in, self.tf_w = tile_factor, tile_factor
        self.fp_max = True
        self.fp_sum = True
        self.fp_target = True
        self.log_softmax = True
        self.reduction = reduction
        assert self.reduction in ['sum', 'mean']
        self.margin = margin
        self.scale = scale
        self.trigger = BackwardTrigger(self.proj_weight)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None

    def get_target_nlprob(self, i: 'torch.Tensor', w: 'torch.Tensor', target: 'torch.Tensor', debase_max: 'torch.Tensor', exp_sums: 'torch.Tensor') ->torch.Tensor:
        """Get target's negative log probability."""
        target_score = TargetScoreFunction.apply(i, w, target, self)
        prob = (target_score - debase_max).exp() / exp_sums
        if self.log_softmax:
            prob = prob.log()
        return -prob.sum()

    def eval_forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """Eval time forward that doesn't fuse the softmax and NLL Loss kernels."""
        return torch.matmul(input, self.proj_weight.T)

    def forward(self, input: 'torch.Tensor', target: 'Optional[torch.Tensor]') ->torch.Tensor:
        if not self.training and target is None:
            return self.eval_forward(input)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            cur_mem = round(torch.cuda.memory_allocated() / 1024 / 1024)
            mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
            None
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if torch.is_grad_enabled():
            assert input.requires_grad
        input, target = _reshape_inputs(input, target)
        tokens, d_model = input.shape
        t2, = target.shape
        vocab, d2 = self.proj_weight.shape
        assert d_model == d2, f'incorrect shape {d_model} vs {d2}'
        assert tokens == t2, f'incorrect shape {tokens} vs {t2}'
        split_dim = 0
        input_split_size = _next_power_of_2_or_max(tokens // self.tf_in, tokens)
        weight_split_size = _next_power_of_2_or_max(vocab // self.tf_w, vocab)
        inputs = torch.split(input, input_split_size, split_dim)
        weight = self.trigger()
        weights = torch.split(weight, weight_split_size, split_dim)
        targets = tuple([torch.Tensor()] * len(inputs))
        if self.scale is not None:
            targets = torch.split(target, input_split_size, split_dim)
        maxs = []
        for i, tgt in zip(inputs, targets):
            m = None
            for w_idx, w in enumerate(weights):
                _m = GetMaxFunction.apply(i, w, tgt, self, w_idx, weight_split_size, split_dim)
                if m is None:
                    m = _m
                else:
                    m = torch.max(m, _m)
            assert m is not None
            maxs.append(m)
        maxs_tensor = torch.cat(maxs)
        assert maxs_tensor.shape == (tokens,)
        sums = []
        for i, tgt, debase_max in zip(inputs, targets, maxs):
            s = None
            for w_idx, w in enumerate(weights):
                _s = GetSumFunction.apply(i, w, tgt, debase_max, self, w_idx, weight_split_size, split_dim)
                if s is None:
                    s = _s
                else:
                    s += _s
            assert s is not None
            sums.append(s)
        sums_tensor = torch.cat(sums)
        assert sums_tensor.shape == (tokens,)
        result = self.get_target_nlprob(input, self.proj_weight, target, maxs_tensor, sums_tensor)
        if self.reduction == 'mean':
            result /= tokens
        return result


class ModelShard(nn.Module):
    """
    Wrap one shard of the model, make it possible to load parameters on the
    fly for the FW and BW pass on the given device.
    """

    def __init__(self, cpu_model_shard: 'nn.Module', device: 'torch.device', offload_device: 'torch.device', index: 'int'):
        super().__init__()
        self.model_shard = cpu_model_shard
        self.index = index
        self.device = device
        torch.device(self.device)
        self.offload_device = offload_device
        self.model_shard
        self._cpu_to_gpu_stream = torch.Stream(device=self.device)
        self._gpu_to_cpu_stream = torch.Stream(device=self.device)

    def forward(self, *inputs):
        return self.model_shard(*inputs) if isinstance(inputs, tuple) else self.model_shard(inputs)

    def to(self, device: 'torch.device') ->'ModelShard':
        self.model_shard
        return self

    def train(self, mode: 'bool'=True) ->'ModelShard':
        self.model_shard.train(mode)
        return self

    def to_device(self) ->None:
        self.model_shard

    def forward_load(self, non_blocking: 'bool'=True) ->None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard

    def backward_load(self, non_blocking: 'bool'=True) ->None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard

    def forward_drop(self, non_blocking: 'bool'=True) ->None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard

    def backward_drop(self, non_blocking: 'bool'=True) ->None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard


def _conditional_amp_bwd_decorator(orig_func):
    if hasattr(torch.amp, 'custom_bwd'):
        return torch.amp.custom_bwd(orig_func)

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) ->Any:
        return orig_func(*args, **kwargs)
    return inner_decorator


def _conditional_amp_fwd_decorator(orig_func):
    if hasattr(torch.amp, 'custom_fwd'):
        return torch.amp.custom_fwd(orig_func)

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) ->Any:
        return orig_func(*args, **kwargs)
    return inner_decorator


_MODEL_PARALLEL_GROUP = None


def get_model_parallel_group() ->torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def ensure_divisibility(numerator: 'int', denominator: 'int') ->None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide_and_check_no_remainder(numerator: 'int', denominator: 'int') ->int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor: 'torch.Tensor', num_partitions: 'int', contiguous_split_chunks: 'bool'=False) ->Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide_and_check_no_remainder(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _split(input_: 'torch.Tensor') ->torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()
    return output


class OffloadModel(nn.Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train by offloading majority of the model parameters to the CPU.
    `OffloadModel` is heavily inspired by the _L2L algorithm and _Zero-Offload.
    ::

        model = get_model()
        offload_model = OffloadModel(model, device,
                                    offload_device=torch.device(“cpu”),
                                    num_slices=3,
                                    checkpoint_activation=True,
                                    num_microbatches=5)

    .. _L2L: https://arxiv.org/abs/2002.05645
    .. _Zero-Offload: https://arxiv.org/abs/2101.06840

    At each step, a layer(or series of layers) are loaded
    onto the GPU for the forward and backward pass with intermediate
    activations being copied onto the GPU as required. Once the forward
    or backward pass is completed for a given shard, it is moved back to
    the CPU again.

    `OffloadModel` supports activation checkpointing which reduces
    the memory footprint. You can also increase the number of
    microbatches which translates to more computation cycles for
    every shard load. This helps offset the cost of moving the shard
    from the CPU to GPU and vice versa.

    Note: OffloadModel currently only supports nn.Sequential models.

    Args:
        module (~torch.nn.Sequential): Module to be offloaded.

        device (torch.device):
            Device where the active model should reside.

        offload_device (torch.device):
            Device where the inactive model should reside.

        num_slices (int):
            Number of slices into which the model should be chunked.

        checkpoint_activation (bool):
            Boolean to indicate if we want to checkpoint intermediate
            activation states on the CPU. Default value is False.

        num_microbatches (int):
            Number of microbatches which should be run per model
            shard on device.
    """

    def __init__(self, model: 'Any', device: 'torch.device', offload_device: 'torch.device'=torch.device('cpu'), num_slices: 'int'=3, checkpoint_activation: 'bool'=False, num_microbatches: 'int'=1):
        super().__init__()
        if not model:
            raise TypeError('`model` argument to `OffloadModel` cannot be None.')
        if not device:
            raise TypeError('`device` argument to `OffloadModel` cannot be None.')
        if not (isinstance(model, nn.Sequential) or type(model) == list):
            raise TypeError('`model` argument to `OffloadModel` must be of type `nn.Sequential`.')
        if not torch.cuda.is_available():
            raise TypeError('CUDA must be available as one of the compute devices for `OffloadModel`.')
        self.device = device
        self.offload_device = offload_device
        self.model_slices: 'List[nn.Module]' = []
        if type(model) == list:
            for i, m in enumerate(model):
                self.model_slices.append(ModelShard(cpu_model_shard=m, device=device, offload_device=offload_device, index=i))
        else:
            splits = _split(model, num_slices)
            for i, split in enumerate(splits):
                self.model_slices.append(ModelShard(cpu_model_shard=nn.Sequential(*split), device=device, offload_device=offload_device, index=i))
        self._model = torch.nn.Sequential(*self.model_slices)
        self._activations: 'List[Tuple]' = []
        if not checkpoint_activation and num_microbatches > 1:
            raise RuntimeError('We currently only support microbatches with activation checkpointing.')
        self._checkpoint_activation = checkpoint_activation
        self._num_microbatches = num_microbatches

    def forward(self, *inputs: Any, **_: Any) ->Any:
        if self._checkpoint_activation:
            return OffloadFunction.apply(*inputs, torch.tensor([], requires_grad=True), self)
        self._activations = []
        for index in range(-1, len(self.model_slices)):
            if index >= 0:
                self._activations[index] = tuple([a for a in list(self._activations[index])])
                inputs = self._activations[index]
                inputs = self.model_slices[index](*inputs)
            inputs = ShardSyncLayer.apply(inputs, index, self.model_slices, self)
            self._activations.append(inputs)
            if index >= 0:
                self._activations[index] = tuple([a.cpu() for a in list(self._activations[index])])
        result = self._activations[-1]
        result = tuple([r for r in result])
        return result[0] if len(result) == 1 else result


def _forward(input: 'Tensor', affine: 'bool', mean: 'Tensor', invstd: 'Tensor', weight: 'Tensor', bias: 'Tensor') ->Tensor:
    if affine:
        return (input - mean) * (invstd * weight.reshape_as(mean)) + bias.reshape_as(mean)
    else:
        return (input - mean) * invstd


class _SyncBatchNormFunction(torch.autograd.Function):
    """
    An autograd function used to avoid storing activations for intermediate results.

    NOTE: Even though the mean and var are passed into this function, we do the entire
    backward, including mean and var, here. We have to calculate statistics outside
    this function in order to avoid multiple all_reduces when using checkpointing.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, affine, mean, invstd, total_count, process_group):
        ctx.save_for_backward(input, weight, bias, mean, invstd, total_count)
        ctx.process_group = process_group
        return _forward(input, affine, mean, invstd, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]
        grad_input = None
        grad_weight = None
        grad_bias = None
        input, weight, bias, mean, invstd, total_count = ctx.saved_tensors
        process_group = ctx.process_group
        dim = [d for d in range(input.ndim) if d != 1]
        if needs_input_grad or needs_weight_grad:
            grad_common = torch.sum((input - mean) * grad_output, dim=dim, keepdim=True)
        if needs_input_grad:
            if weight is None:
                grad_input = invstd * grad_output
                grad_mean = -torch.sum(grad_input, dim=dim, keepdim=True)
                grad_invstd = grad_common
            else:
                grad_input = invstd * weight.reshape_as(mean) * grad_output
                grad_mean = -torch.sum(grad_input, dim=dim, keepdim=True)
                grad_invstd = grad_common * weight.reshape_as(mean)
            grad_var = -0.5 * invstd.pow(3) * grad_invstd
            grad_mean += -2 * mean * grad_var
            grad_meansqr = grad_var
            vec = torch.cat([grad_mean, grad_meansqr])
            all_reduce_handle = dist.all_reduce(vec, group=process_group, async_op=True)
        if needs_weight_grad:
            grad_weight = (grad_common * invstd).resize_as(weight)
            grad_bias = torch.sum(grad_output, dim=dim)
        if needs_input_grad:
            all_reduce_handle.wait()
            vec = vec / total_count
            grad_mean, grad_meansqr = vec.chunk(2)
            grad_input += grad_mean
            grad_input += input * (2 * grad_meansqr)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


def _calculate_stats(input: 'Tensor', eps: 'float', process_group: 'ProcessGroup') ->Tuple[Tensor, Tensor, Tensor, Tensor]:
    dim = [d for d in range(input.ndim) if d != 1]
    count = torch.full((1,), input.numel() // input.size(1), device=input.device, dtype=input.dtype)
    total_count = count.clone()
    all_reduce_handle = dist.all_reduce(total_count, group=process_group, async_op=True)
    mean = torch.mean(input, dim=dim, keepdim=True)
    meansqr = torch.mean(input * input, dim=dim, keepdim=True)
    vec = torch.cat([mean, meansqr])
    all_reduce_handle.wait()
    vec = vec * (count / total_count)
    dist.all_reduce(vec, group=process_group)
    mean, meansqr = vec.chunk(2)
    var = meansqr - mean * mean
    invstd = torch.rsqrt(var + eps)
    return mean, var, invstd, total_count


def _track_running_stats(running_mean: 'Tensor', running_var: 'Tensor', momentum: 'float', mean: 'Tensor', var: 'Tensor', total_count: 'Tensor') ->None:
    unbiased_var = var * (total_count / (total_count - 1))
    running_mean += momentum * (mean.reshape(-1) - running_mean)
    running_var += momentum * (unbiased_var.reshape(-1) - running_var)


def is_checkpointing() ->bool:
    """Whether the current forward propagation is under checkpointing.

    Returns:
        bool: :data:`True` if it's under checkpointing.

    """
    return thread_local.is_checkpointing


def is_recomputing() ->bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.

    .. seealso:: :ref:`Detecting Recomputation`

    """
    return thread_local.is_recomputing


class SyncBatchNorm(torch.nn.BatchNorm2d):
    """
    Fast re-implementation of ``torch.nn.SyncBatchNorm`` that can achieve a speedup
    of 5x or more over the default implementation depending on size of the input
    and number of distributed workers.
    """

    def __init__(self, *args: Tuple[Any, ...], process_group: Optional[ProcessGroup]=None, **kwargs: Dict[str, Any]) ->None:
        super().__init__(*args, **kwargs)
        self._process_group = process_group if process_group is not None else dist.group.WORLD
        self.saved_for_2nd_fwd: 'List[Tuple]' = []
        self.disable_patch_batchnorm = True

    def forward(self, input: 'Tensor') ->Tensor:
        if not dist.is_initialized() or not self.training:
            return super().forward(input)
        wrapped = is_checkpointing() or is_recomputing()
        if not wrapped or is_checkpointing():
            with torch.no_grad():
                mean, var, invstd, total_count = _calculate_stats(input, self.eps, self._process_group)
                if self.track_running_stats:
                    _track_running_stats(self.running_mean, self.running_var, self.momentum, mean, var, total_count)
        if is_checkpointing():
            self.saved_for_2nd_fwd.append((mean, invstd, total_count))
            return _forward(input, self.affine, mean, invstd, self.weight, self.bias)
        if is_recomputing():
            mean, invstd, total_count = self.saved_for_2nd_fwd.pop(0)
        return _SyncBatchNormFunction.apply(input, self.weight, self.bias, self.affine, mean, invstd, total_count, self._process_group)

    @classmethod
    def convert_sync_batchnorm(cls, module: 'torch.nn.Module', process_group: 'Optional[ProcessGroup]'=None) ->torch.nn.Module:
        """Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`fairscale.experimental.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = fairscale.experimental.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, process_group=process_group)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, 'qconfig'):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


class FlatParameter(nn.Parameter):
    """A parameter that is initialized from a list of parameters and can be
    turned into a list of views as needed.
    """

    def __new__(cls, params: 'Sequence[nn.Parameter]', requires_grad: 'bool'=True) ->'FlatParameter':
        """Make an object using the parent's __new__ function."""
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError('An non-empty list or tuple argument is needed')
        if not all(isinstance(p, (nn.Parameter, Tensor)) for p in params):
            raise ValueError('List items need to be Parameter types')
        if any(isinstance(p, FlatParameter) for p in params):
            raise ValueError('Nesting FlatParameter is not supported')
        data = torch.cat([(p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1)) for p in params], 0)
        return super(FlatParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, params: 'Sequence[nn.Parameter]', requires_grad: 'bool'=True):
        """Initialize the _param_numels and _param_shapes lists."""
        self._param_numels = [p.numel() for p in params]
        assert self.numel() <= sum(self._param_numels), f'Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}'
        self._param_shapes = [p.size() for p in params]
        self._param_infos: 'List[Tuple[str, nn.Module, str]]' = []
        self._shared_param_infos: 'List[Tuple[str, str, nn.Module, str, nn.Module, str]]' = []

    def get_param_views(self, external_data: 'Optional[Tensor]'=None) ->Iterator[Tensor]:
        """Return a generator of views that map to the original parameters."""
        assert self.data.numel() <= sum(self._param_numels), f'Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}'
        data = external_data if external_data is not None else self
        if data.numel() != sum(self._param_numels):
            raise ValueError(f'Incorrect numel of supplied data: got {data.numel()} but expected {sum(self._param_numels)}')
        return (t.view(s) for t, s in zip(data.split(self._param_numels), self._param_shapes))

    def metadata(self) ->Tuple[List[str], List[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for this flat parameter."""
        names = [('.'.join([m, n]) if m else n) for m, _, n in self._param_infos]
        return names, self._param_shapes, self._param_numels

    def __setstate__(self, state: 'Tuple[Any, Any, Any, Any]') ->None:
        """Use by pickle to set the internal states."""
        self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos = state
        assert self.numel() <= sum(self._param_numels), f'Incorrect pickling {self.numel()} vs. {sum(self._param_numels)}'

    def __reduce_ex__(self, proto: 'int') ->Tuple[Any, Any, Any]:
        """Support pickling between ranks."""
        return FlatParameter, ([self.data], self.requires_grad), (self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos)


def replace_by_prefix_(state_dict: "Union[Dict[str, Tensor], 'OrderedDict[str, Tensor]']", old_prefix: 'str', new_prefix: 'str') ->None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError('old_prefix and new_prefix must be distinct')
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix):]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _post_state_dict_hook(module: 'nn.Module', state_dict: "'OrderedDict[str, Tensor]'", prefix: 'str', *args: Any) ->'OrderedDict[str, Tensor]':
    replace_by_prefix_(state_dict, prefix + '_fpw_module.', prefix)
    return state_dict


_enable_pre_load_state_dict_hook = True


def _pre_load_state_dict_hook(state_dict: "Union[Dict[str, Tensor], 'OrderedDict[str, Tensor]']", prefix: 'str', *args: Any) ->None:
    if not _enable_pre_load_state_dict_hook:
        return
    replace_by_prefix_(state_dict, prefix, prefix + '_fpw_module.')
    flat_param_key = prefix + '_fpw_module.flat_param'
    for k in list(state_dict.keys()):
        if k.startswith(flat_param_key):
            last_part = k.split('.')[-1]
            assert last_part.startswith('flat_param_'), last_part
            replace_by_prefix_(state_dict, k, prefix + last_part)


class ProcessGroupName(str, Enum):
    default = 'default'
    reduce_scatter = 'reduce_scatter'


class Bucket:
    """
    Helper class to simplify the handling of buckets, which unify the underlying storage of multiple tensors
    """

    def __init__(self, size: 'int', dtype: 'torch.dtype', device: 'torch.device') ->None:
        self._params: 'List[torch.Tensor]' = []
        self._param_ids: 'List[int]' = []
        self._fill = 0
        self.buffer: 'torch.Tensor' = torch.zeros(size, dtype=dtype, device=device)

    def to(self, device: 'Optional[Union[int, torch.device]]', dtype: 'Optional[torch.dtype]'=None, non_blocking: 'bool'=False, keep_param_alignment: 'bool'=True) ->'ParamBucket':
        """
        Move the underlying buffer
        """
        assert self.buffer is not None, 'Cannot move a collapsed bucket, please rebuild it'
        self.buffer = self.buffer


class ReduceScatterBucketer:
    """
    Helper for bucketing multiple reduce-scatter operations on small tensors
    into larger reduce-scatter ops to improve communication efficiency.

    Usage::

        bucketer = ReduceScatterBucketer()
        bucketer.reduce_scatter_async(
            small_tensors, callback_fn=lambda result: print("small")
        )
        bucketer.reduce_scatter_async(
            big_tensors, callback_fn=lambda result: print("big")
        )
        bucketer.reduce_scatter_async(
            more_small_tensors, callback_fn=lambda result: print("small2")
        )
        bucketer.flush()  # callbacks only guaranteed to be called after flush()
        # Example output (note that it is out of order, due to bucketing):
        # big
        # small
        # small2

    Args:
        bucket_cap_mb (int, Optional): bucket size for communicating. Buckets
            are sub-divided based on world_size. Values <= 0 disable bucketing.
    """

    def __init__(self, bucket_cap_mb: 'int'=25):
        self.bucket_cap_mb = bucket_cap_mb
        self.buckets: "Dict[Tuple[torch.dtype, torch.device, 'ProcessGroup'], Bucket]" = {}

    @torch.no_grad()
    def reduce_scatter_async(self, input_list: 'List[Tensor]', group: "'ProcessGroup'", callback_fn: 'Optional[Callable]'=None) ->None:
        """
        Reduce-scatter a list of tensors asynchronously, so smaller reductions
        can be bucketed together. The given callback (``callback_fn``) will be
        called with the reduced result at some later time. Call ``flush()`` to
        force all queued ops and callbacks to be executed.

        Note that large inputs will be reduced immediately, and this function
        may also flush the relevant bucket to make room for ``input_list``.

        Args:
            input_list (List[Tensor]): list of tensors to reduce-scatter. List
                should contain ``group.size()`` tensors and each tensor should
                have identical shape, dtype and device.
            group (ProcessGroup): process group for reduction
            callback_fn (Callable, Optional): callback function to call after
                the reduction executes. Function will be called with a single
                argument corresponding to the reduced result.
        """
        world_size = group.size()
        assert len(input_list) == world_size, f'reduce_scatter received {len(input_list)} inputs, expected group.size() ({world_size})'
        first_input = input_list[0]
        first_input_size = first_input.numel()
        bucket_shard_size = self._get_shard_size(first_input.element_size(), world_size)
        if first_input_size > bucket_shard_size:
            output = torch.zeros_like(input_list[0])
            if hasattr(dist, '_reduce_scatter_base') and enable_nccl_base_collectives:
                input_flattened = torch.cat(input_list)
                dist._reduce_scatter_base(output, input_flattened, group=group)
            else:
                dist.reduce_scatter(output, input_list, group=group)
            if callback_fn is not None:
                callback_fn(output)
            return
        bucket = self._get_bucket(first_input, group)
        if first_input_size > bucket.data.size(1) - bucket.offset:
            bucket.flush()
        stacked_input = torch.stack(input_list).view(world_size, first_input_size)
        offset = bucket.offset
        bucket.data[:, offset:offset + first_input_size].copy_(stacked_input)
        bucket.offset += first_input_size
        if callback_fn is not None:
            result_view = bucket.output_shard[offset:offset + first_input_size].view_as(first_input)
            bucket.callbacks.append(functools.partial(callback_fn, result_view))

    @torch.no_grad()
    def flush(self) ->None:
        """Reduce-scatter any partial buckets."""
        for bucket in self.buckets.values():
            bucket.flush()

    @torch.no_grad()
    def teardown(self) ->None:
        """Free buffers from all buckets."""
        for bucket in self.buckets.values():
            bucket.teardown()

    @functools.lru_cache()
    def _get_shard_size(self, element_size: 'int', num_shards: 'int') ->int:
        if self.bucket_cap_mb <= 0:
            return 0
        MB = 1024 * 1024
        bucket_size = self.bucket_cap_mb * MB / element_size
        return int(bucket_size // num_shards)

    def _get_bucket(self, tensor: 'Tensor', group: "'ProcessGroup'") ->Bucket:
        key = tensor.dtype, tensor.device, group
        if key not in self.buckets:
            world_size = group.size()
            shard_size = self._get_shard_size(tensor.element_size(), world_size)
            data = tensor.new_zeros((world_size, shard_size))
            self.buckets[key] = Bucket(data, group)
        self.buckets[key].setup()
        return self.buckets[key]


class TrainingState(Enum):
    """
    Simple enum to indicate what state FSDP is in. Used for asserting
    to make sure APIs are called in the correct state.

    ..note::

        BACKWARD_PRE and BACKWARD_POST states are used to ensure we
        receives backward hooks in the correct order. It is used to catch
        unexpected order of hooks being called (likely due to our
        hook registration logic or autograd engine logic changes).

    TODO (Min): It would be nice to capture the stepping state as well.
        Maybe we can use the model.zero_grad() call, but not sure if it
        is called if optim.zero_grad() is used instead.
        It would be nice to have clear state transition be explicit like:

        zero_grad -> fwd -> bwd -> optionally accum grad by repeating
        fwd/bwd -> stepping -> loop back to zero_grad
    """
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


def _clean_path(path: 'str') ->str:
    """Remove FSDP related wrapper modules from a given state dict key str path."""
    return '.'.join([split for split in path.split('.') if split not in {'_fsdp_wrapped_module', '_fpw_module'}])


def _get_default_cuda_device(module: 'nn.Module') ->torch.device:
    """Try to infer CUDA device from module parameters."""
    try:
        compute_device = next(module.parameters()).device
        if compute_device.type == 'cuda':
            return compute_device
    except StopIteration:
        pass
    return torch.device('cuda')


def _unpad(shard: 'torch.Tensor', pad: 'int') ->torch.Tensor:
    if pad > 0:
        shard = shard[:-pad]
    return shard


@torch.no_grad()
def alloc_storage_(data: 'torch.Tensor', size: 'torch.Size') ->None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():
        return
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())


def apply_to_type(type_fn: 'Callable', fn: 'Callable', container: 'Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set, NamedTuple]') ->Any:
    """Recursively apply to all objects in different kinds of container types that matches a type function."""

    def _apply(x: 'Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set]') ->Any:
        if type_fn(x):
            return fn(x)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = _apply(value)
            return od
        elif isinstance(x, PackedSequence):
            _apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            f = getattr(x, '_fields', None)
            if f is None:
                return tuple(_apply(x) for x in x)
            else:
                assert isinstance(f, tuple), 'This needs to be a namedtuple'
                x = cast(NamedTuple, x)
                _dict: 'Dict[str, Any]' = x._asdict()
                _dict = {key: _apply(value) for key, value in _dict.items()}
                return type(x)(**_dict)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    return _apply(container)


def apply_to_tensors(fn: 'Callable', container: 'Union[torch.Tensor, Dict, List, Tuple, Set]') ->Any:
    """Recursively apply to all tensor in different kinds of container types."""
    return apply_to_type(torch.is_tensor, fn, container)


def calc_grad_norm(parameters: 'List[torch.nn.Parameter]', p: 'float') ->torch.Tensor:
    """Calculate gradient norm of an iterable of parameters.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda par: par.grad is not None, parameters))
    if len(parameters) == 0:
        return torch.tensor(0.0)
    p = float(p)
    if p == inf:
        local_norm = max(par.grad.detach().abs().max() for par in parameters)
    else:
        local_norm = torch.norm(torch.stack([torch.norm(par.grad.detach(), p, dtype=torch.float32) for par in parameters]), p)
    return local_norm


def cast_floats_to_right_precision(to_fp16: 'bool', no_grad: 'bool', *args: Any, **kwargs: Any) ->Tuple[Any, Any]:
    """
    Cast floating point Tensors in *args or **kwargs to FP16 or FP32 if they are not.
    We also retain the requires_grad flag so that casting doesn't affect the autograd graph.
    """

    def fn_fp16(x: 'torch.Tensor') ->torch.Tensor:
        if x.dtype is torch.float32:
            y = x.half()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x

    def fn_fp32(x: 'torch.Tensor') ->torch.Tensor:
        if x.dtype is torch.float16:
            y = x.float()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x
    fn = fn_fp16 if to_fp16 else fn_fp32
    context = torch.no_grad() if no_grad else contextlib.suppress()
    with context:
        return apply_to_tensors(fn, args), apply_to_tensors(fn, kwargs)


def chunk_and_pad(tensor: 'torch.Tensor', num_chunks: 'int') ->List[torch.Tensor]:
    """Chunk a given Tensor into num_chunks parts and add any necessary padding."""
    chunks = list(torch.flatten(tensor).chunk(num_chunks))
    num_pad_for_partial_chunk = chunks[0].numel() - chunks[-1].numel()
    if num_pad_for_partial_chunk > 0:
        chunks[-1] = F.pad(chunks[-1], [0, num_pad_for_partial_chunk])
    if len(chunks) < num_chunks:
        chunks.extend([torch.zeros_like(chunks[0]) for _ in range(num_chunks - len(chunks))])
    return chunks


def enable_pytorch_sync_bn(module: 'torch.nn.Module') ->None:
    """Call _specify_ddp_gpu_num for all pytorch SyncBN layers so that it
    is happily running even without DDP. E.g. this is used by FSDP.
    """
    for layer in module.modules():
        if isinstance(layer, torch.nn.modules.SyncBatchNorm) and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)


def free_storage_(data: 'torch.Tensor') ->None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        assert data.storage_offset() == 0
        data.storage().resize_(0)

