import sys
_module = sys.modules[__name__]
del sys
conf = _module
simple_sampling = _module
ggmltensor = _module
configurator = _module
export = _module
model = _module
sample = _module
tinystories = _module
tokenizer = _module
train = _module
bisect_nvfuser = _module
validate_build = _module
setup = _module
__about__ = _module
thunder = _module
benchmarks = _module
benchmark_litgpt = _module
conftest = _module
distributed = _module
einsum = _module
targets = _module
test_benchmark_litgpt = _module
clang = _module
langctx = _module
common = _module
core = _module
baseutils = _module
codeutils = _module
compile_data = _module
devices = _module
dtypes = _module
functionalization = _module
interpreter = _module
jit_ext = _module
langctxs = _module
module = _module
options = _module
patterns = _module
prims = _module
profile = _module
proxies = _module
pytree = _module
recipe = _module
rematerialization = _module
symbol = _module
trace = _module
trace_interpreter = _module
transform_common = _module
transforms = _module
utils = _module
vjp_utils = _module
dev_utils = _module
debug_transform = _module
nvtx_profile_transform = _module
utils = _module
distributed = _module
bucketing = _module
checkpoint = _module
prims = _module
tensor_parallel = _module
column_wise = _module
common = _module
optimize_comm = _module
row_wise = _module
transforms = _module
ddp = _module
ddp_v2 = _module
fsdp = _module
fsdp_v2 = _module
utils = _module
dynamo = _module
compiler = _module
compiler_graph_benchmark = _module
report = _module
splitter = _module
utils = _module
examine = _module
memory_calculation = _module
executors = _module
apex_entropyex_impl = _module
apex_fused_rms_norm_impl = _module
apexex = _module
cudnn_layernormex = _module
cudnnex = _module
fa3ex = _module
nvfuserex = _module
nvfuserex_impl = _module
passes = _module
pythonex = _module
sdpaex = _module
torch_autograd = _module
torch_compile = _module
torchex = _module
transformer_engineex = _module
triton_crossentropy = _module
triton_crossentropy_impl = _module
triton_utils = _module
utils = _module
extend = _module
numpy = _module
langctx = _module
recipes = _module
hf_bert = _module
tests = _module
bf16 = _module
helper = _module
modules = _module
test_checkpoint = _module
test_ddp = _module
test_fsdp = _module
test_ops = _module
test_tensor_parallel = _module
framework = _module
hf_bart_self_attn = _module
litgpt_model = _module
llama2_model = _module
make_tensor = _module
module_example = _module
nanogpt_model = _module
opinfos = _module
test_apex_cross_entropy_executor = _module
test_apex_fused_norms = _module
test_auto_register_torchops = _module
test_autocast = _module
test_core = _module
test_cudnn_executor = _module
test_dynamo = _module
test_einops = _module
test_elementwise = _module
test_examine_memory = _module
test_extend = _module
test_fa3_executor = _module
test_grad = _module
test_inplace_copy = _module
test_inplace_functionalization = _module
test_interpreter = _module
test_jit_general = _module
test_networks = _module
test_nvfuser = _module
test_nvfuser_remat = _module
test_ops = _module
test_patterns = _module
test_pythonex = _module
test_randomness = _module
test_recipes = _module
test_reductions = _module
test_sdpaex_executor = _module
test_shape_ops = _module
test_torch_compile_executor = _module
test_transformer_engine_executor = _module
test_transforms = _module
test_triton_ce = _module
default_torch_ops = _module
langctx = _module
autocast = _module
constant_folding = _module
cudagraph = _module
extraction_only_prologue_transform = _module
materialization = _module
prune_prologue_checks = _module
qlora = _module
quantization = _module

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


import inspect


import torch


import math


from typing import List


import numpy as np


import collections


from enum import Enum


import functools


import numpy


from torch import nn


from typing import Any


from typing import Optional


from typing import Tuple


import torch.nn.functional as F


from functools import partial


import random


import torch.distributed as dist


import time


from torch.distributed import destroy_process_group


from torch.distributed import init_process_group


from torch.nn.parallel import DistributedDataParallel as DDP


from functools import wraps


from collections import defaultdict


from collections import namedtuple


from collections.abc import Callable


from collections.abc import Sequence


import warnings


import torch as pytorch


from collections import UserDict


from numbers import Number


import torch.multiprocessing as mp


import torch.nn as nn


from torch.testing import make_tensor


from typing import TYPE_CHECKING


from torch.utils.data import DataLoader


from torch.utils.data import IterableDataset


import torch.distributed as torch_dist


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper


from enum import auto


import pandas as pd


from functools import reduce


from types import EllipsisType


from typing import Union


from collections.abc import Generator


from collections.abc import Hashable


from collections import deque


import torch as torch


from types import MappingProxyType


from types import ModuleType


from types import CodeType


from types import FunctionType


from types import MethodType


import collections.abc


import re


from inspect import Parameter


from typing import NamedTuple


from collections.abc import Iterable


import enum


from typing import Literal


from typing import TypedDict


from collections.abc import Iterator


from collections.abc import Mapping


from collections.abc import MutableMapping


from collections.abc import Set


from collections.abc import Sized


from types import CellType


from types import ClassMethodDescriptorType


from types import CoroutineType


from types import FrameType


from types import MethodDescriptorType


from types import NoneType


from types import BuiltinFunctionType


from types import BuiltinMethodType


from types import MethodWrapperType


from types import WrapperDescriptorType


from types import TracebackType


from types import GetSetDescriptorType


from types import UnionType


import torch.utils.checkpoint


import itertools


from torch.utils.weak import WeakTensorKeyDictionary


from typing import Type


from typing import Dict


import copy


from itertools import chain


import string


from abc import ABC


from itertools import filterfalse


from itertools import compress


from functools import lru_cache


from typing import overload


from typing import Generic


from typing import TypeVar


from inspect import Signature


from torch.utils.benchmark import Timer


import torch.distributed as tdist


from typing import TypeGuard


from torch import Tensor


from torch.distributed._tensor import DTensor


from torch.distributed._tensor import Shard


from torch.nn import Module


from abc import abstractmethod


import torch.distributed


from typing import ClassVar


from torch.distributed import distributed_c10d


from torch.distributed import ProcessGroup


from torch.fx.passes.split_module import split_module


from torch.nn.modules.module import _addindent


from torch._subclasses.fake_tensor import FakeTensor


from warnings import warn


from collections import OrderedDict


from typing import Set


from copy import copy


from collections.abc import Collection


from collections.abc import MutableSet


from collections.abc import MutableSequence


from torch._subclasses.fake_tensor import FakeTensorMode


import torch.cuda


from torch.distributed import distributed_c10d as c10d


from torch.testing._internal import common_utils


from itertools import product


from torch.testing import assert_close


from torch.distributed.fsdp import FullyShardedDataParallel


from torch.distributed.fsdp.wrap import always_wrap_policy


from functools import singledispatchmethod


from torch._dynamo import is_inductor_supported


from typing import cast


from torch.nn import functional as F


from torch.distributed import is_available


from itertools import islice


from torch._dynamo.eval_frame import is_inductor_supported


import torch.fx


import torch.testing


import torch._higher_order_ops.wrap


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: 'int', eps: 'float'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cos: 'torch.Tensor', freqs_sin: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor]:
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    a, b = freqs_cos.shape
    freqs_cos = freqs_cos.view(1, a, 1, b)
    freqs_sin = freqs_sin.view(1, a, 1, b)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: 'torch.Tensor', n_rep: 'int') ->torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):

    def __init__(self, args: 'ModelArgs'):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

    def forward(self, x: 'torch.Tensor', freqs_cos: 'torch.Tensor', freqs_sin: 'torch.Tensor'):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int', dropout: 'float'):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: 'int', args: 'ModelArgs'):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim, multiple_of=args.multiple_of, dropout=args.dropout)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = 0,
    COMPLEX_TO_FLOAT = 1,
    KEEP_PROMOTED_TYPE = 2,
    ALWAYS_BOOL = 3,


class DistParallelType(Enum):
    NONE = auto()
    REPLICATED = auto()
    FULLY_SHARDED = auto()
    COLUMN_WISE = auto()
    ROW_WISE = auto()

