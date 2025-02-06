import sys
_module = sys.modules[__name__]
del sys
prepare_evaluation_data = _module
lcm = _module
datasets = _module
base = _module
batch = _module
configs = _module
dataloader = _module
dataloading = _module
parquet_utils = _module
sentence_splitter_pipeline = _module
sentence_splitting = _module
utils = _module
evaluation = _module
api = _module
arun = _module
cli = _module
configs = _module
local = _module
params = _module
slurm = _module
metrics = _module
coherence_metrics = _module
common = _module
multilingual_similarity = _module
round_trip = _module
seahorse = _module
sentence_fluency = _module
similarity = _module
predictors = _module
dummy = _module
gemma = _module
huggingface = _module
lcm = _module
llama = _module
two_tower_diffusion_lcm = _module
run = _module
tasks = _module
base = _module
cnn_dailymail = _module
dummy = _module
lcm_generation = _module
xlsum = _module
xsum = _module
common = _module
data_utils = _module
distributed = _module
hf = _module
segment_alignment = _module
sonar = _module
generator = _module
scorer = _module
generator = _module
scorer = _module
models = _module
abstract_lcm = _module
builder = _module
base_lcm = _module
archs = _module
builder = _module
frontend = _module
loader = _module
normalization = _module
sonar_normalizer = _module
builder = _module
builder = _module
frontend = _module
nn = _module
denoisers = _module
attention_masks = _module
factory = _module
lcm_denoiser = _module
incremental_state = _module
initialization = _module
normalization = _module
projection = _module
schedulers = _module
ddim = _module
timestep_encoder = _module
transformer = _module
attention = _module
decoder = _module
factory = _module
criterion = _module
criterion = _module
trainer = _module
metrics = _module
mse_lcm = _module
criterion = _module
optim = _module
step_sampler = _module
trainer = _module
criterion = _module
card_utils = _module
common = _module
distributed = _module
logging = _module
model_type_registry = _module
fit_embedding_normalizer = _module
prepare_wikipedia = _module
tests = _module
common = _module
test_headers = _module
units = _module
conftest = _module
test_parquet_utils = _module
test_sentence_splitter = _module
test_cli = _module
test_generation_tasks = _module
test_judge_tasks = _module
test_metrics = _module
test_model_based_metrics = _module
test_predictors = _module
test_round_trip = _module
test_similarity = _module
test_task_registry = _module
test_base_lcm_batched_inference = _module
test_base_lcm_kv_caching = _module
test_lcm_architecture = _module
test_recipes = _module
conftest = _module
test_batch = _module
test_get_trainer = _module
test_toy_task_trainer = _module

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


from typing import Literal


from typing import Optional


import numpy as np


import pandas as pd


import torch


from scipy.signal import find_peaks


from abc import ABC


from abc import abstractmethod


from typing import Callable


from typing import Dict


from typing import Generic


from typing import Iterator


from typing import Sequence


from typing import TypeVar


from typing import Union


from copy import deepcopy


from enum import Enum


from typing import Any


from typing import List


from typing import Tuple


from torch import Tensor


from torch.nn import Module


import re


from functools import partial


from typing import Mapping


from functools import lru_cache


from typing import Generator


from functools import reduce


from functools import wraps


from numpy.typing import NDArray


import typing as tp


import abc


from typing import Protocol


from typing import Type


from typing import runtime_checkable


from numpy.random import RandomState


import time


from collections import Counter


from typing import Iterable


from typing import cast


from functools import cached_property


from typing import TYPE_CHECKING


import inspect


from collections import defaultdict


import string


from inspect import Parameter


from itertools import accumulate


from itertools import product


from logging import LogRecord


from itertools import islice


from itertools import zip_longest


from logging import Logger


from types import ModuleType


import torch.distributed as dist


import torch.nn


from torch.nn import Dropout


from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


from typing import final


import math


import torch.nn as nn


from torch.nn import ModuleList


from torch.nn.parameter import Parameter


from torch import Generator


from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.optim import Optimizer


from collections.abc import MutableMapping


from typing import Set


from torch.cuda import _get_device_index


import torch.distributions as D


from itertools import count


from typing import ContextManager


from torch.profiler import record_function


import torch.nn.functional as F


from typing import Sized


import random


import warnings


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from random import randint


from torch.cuda import OutOfMemoryError


class AbstractLCModel(Module):
    """Asbtract Class for LCM models"""

    def __init__(self, config: 'AbstractLCModelConfig') ->None:
        """
        Asbtract LCM model
        """
        super().__init__()
        self.config = config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


logger = logging.getLogger('lcm.evaluation.test_judge')


SONAR_STD = 0.006


def init_linear_kaiming_uniform(layer: 'Linear') ->None:
    torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        fan_in = layer.weight.size(1)
        m = 1
        if layer.weight.ndim > 2:
            for s in layer.weight.shape[2:]:
                m *= s
        fan_in *= m
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(layer.bias, -bound, bound)


def init_linear_to_sonar(layer: 'Linear', sonar_std: 'float') ->None:
    """
    Initialize the post-lcm in such a way, that if it is fed layer-normed
    lcm outputs (with zero mean and unit variance), its outputs have zero
    mean and the variance of SONAR embeddings.
    """
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
    std = sonar_std * (3 / layer.input_dim) ** 0.5
    torch.nn.init.uniform_(layer.weight, a=-std, b=std)


def init_linear_trunc_normal(layer: 'Linear') ->None:
    torch.nn.init.trunc_normal_(layer.weight, std=0.001)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


def init_linear_xavier(layer: 'Linear') ->None:
    torch.nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


def init_linear_zero(layer: 'Linear') ->None:
    torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


def get_init_fn(style: 'str'='xavier', sonar_std: 'float'=SONAR_STD):
    if style == 'xavier':
        return init_linear_xavier
    if style == 'kaiming_uniform':
        return init_linear_kaiming_uniform
    if style == 'sonar':
        return partial(init_linear_to_sonar, sonar_std=sonar_std)
    if style == 'zero':
        return init_linear_zero
    if style == 'trunc_normal':
        return init_linear_trunc_normal
    if style == 'none':
        return None
    else:
        raise ValueError(f'Could not recognize initialization function {style}')


class FFTInterface:

    @staticmethod
    def fft_transform(embeddings: 'Tensor') ->Tensor:
        dtype = embeddings.dtype
        if dtype in [torch.float16, torch.bfloat16]:
            embeddings = embeddings
        embeddings = torch.fft.rfft(embeddings, norm='backward')
        return torch.concat([torch.real(embeddings), torch.imag(embeddings)[..., 1:-1]], dim=-1)

    @staticmethod
    def fft_inverse_transform(embeddings: 'Tensor') ->Tensor:
        assert embeddings.shape[-1] % 2 == 0
        dtype = embeddings.dtype
        if dtype in [torch.float16, torch.bfloat16]:
            embeddings = embeddings
        rr, im = torch.split(embeddings, [embeddings.shape[-1] // 2 + 1, embeddings.shape[-1] // 2 - 1], dim=-1)
        im = torch.concat([torch.zeros_like(im[..., :1]), im, torch.zeros_like(im[..., :1])], dim=-1)
        embeddings = torch.fft.irfft(rr + im * 1.0j)
        return embeddings


class SonarNormalizer(FFTInterface, Module):
    """
    To perform efficient diffusion modeling, SONAR embeddings need to be
    normalized. This SonarNormalizer follows the robust normalization introduced in
    https://arxiv.org/abs/2307.05445
    Quoting from the paper: "Due to the very long-tailed feature distribution, typical mean and standard deviation statistics will be
    heavily biased. We thus propose a robust alternative based on the feature distribution quantiles. We
    take the median as the center of the distribution and approximate its scale using the Normalized
    InterQuartile Range (IQR) for a normal distribution: 0.7413 × IQR
    """

    def __init__(self, config: 'SonarNormalizerConfig', device: 'Optional[Device]'=None, dtype: 'Optional[DataType]'=None) ->None:
        super().__init__()
        self.config = config
        self.register_buffer('center', torch.zeros(config.dim, dtype=dtype, device=device))
        self.register_buffer('scale', torch.ones(config.dim, dtype=dtype, device=device))
        if self.config.clip_proba is not None:
            self.register_buffer('clip_min', torch.ones(config.dim, dtype=dtype, device=device))
            self.register_buffer('clip_max', torch.ones(config.dim, dtype=dtype, device=device))

    def normalize(self, embeddings: 'Tensor') ->Tensor:
        if self.config.with_fft:
            embeddings = self.fft_transform(embeddings)
        embeddings = (embeddings - self.center) / self.scale
        if self.config.clip_proba is not None:
            embeddings = torch.clamp(embeddings, min=self.clip_min, max=self.clip_max)
        return embeddings

    def denormalize(self, embeddings: 'Tensor') ->Tensor:
        if self.config.clip_proba is not None:
            embeddings = torch.clamp(embeddings, min=self.clip_min, max=self.clip_max)
        embeddings = embeddings * self.scale + self.center
        if self.config.with_fft:
            embeddings = self.fft_inverse_transform(embeddings)
        return embeddings

    @torch.no_grad()
    def fit(self, embeddings: 'Tensor'):
        if self.config.normalization_method in ['robust', 'gaussian_robust']:
            from sklearn.preprocessing import RobustScaler
            _scaler = RobustScaler(unit_variance=self.config.normalization_method == 'gaussian_robust', quantile_range=(self.config.quantile_min, self.config.quantile_max))
        elif self.config.normalization_method == 'standard':
            from sklearn.preprocessing import StandardScaler
            _scaler = StandardScaler()
        else:
            raise ValueError(f'Unrecognizable method {self.config.normalization_method} for scaling input features')
        assert embeddings.shape[-1] == self.config.dim
        assert len(embeddings.shape) == 2
        if self.config.with_fft:
            embeddings = self.fft_transform(embeddings)
        embeddings = _scaler.fit_transform(embeddings.cpu().float().numpy())
        if self.config.normalization_method in ['robust', 'gaussian_robust']:
            _center = _scaler.center_
            _scale = _scaler.scale_
        elif self.config.normalization_method == 'standard':
            _center = _scaler.mean_
            _scale = _scaler.scale_
        self.center[:] = torch.tensor(_center, dtype=self.center.dtype, device=self.center.device)
        self.scale[:] = torch.tensor(_scale, dtype=self.scale.dtype, device=self.scale.device)
        if self.config.clip_proba is not None:
            self.clip_min[:] = torch.quantile(torch.tensor(embeddings), self.config.clip_proba, dim=0)
            self.clip_max[:] = torch.quantile(torch.tensor(embeddings), 1 - self.config.clip_proba, dim=0)


class AdaLNModulator(Module):
    """An adaptive LayerNorm modulator to estimate
    shift, gate and scale for all 3 sub-modules."""

    def __init__(self, input_dim: 'int', output_dim: 'int', device: 'Optional[Device]'=None, dtype: 'Optional[DataType]'=None):
        super().__init__()
        self.activate = nn.SiLU()
        self.fc = nn.Linear(input_dim, 9 * output_dim, bias=True, device=device, dtype=dtype)

    def reset_parameters(self):
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, context: 'Tensor') ->Tuple[Tensor, Tensor, Tensor]:
        modulate_san, modulate_cross_attention, modulate_ffn = self.fc(self.activate(context)).chunk(3, dim=-1)
        return modulate_san, modulate_cross_attention, modulate_ffn


def parse_activation_fn(var: 'str'=None) ->Optional[Module]:
    if var is None:
        return None
    activ_fn: 'Module'
    if var == 'relu':
        activ_fn = torch.nn.ReLU()
    elif var == 'tanh':
        activ_fn = torch.nn.Tanh()
    elif var == 'elu':
        activ_fn = torch.nn.ELU()
    elif var == 'leaky_relu':
        activ_fn = torch.nn.LeakyReLU()
    elif var == 'prelu':
        activ_fn = torch.nn.PReLU()
    elif var == 'selu':
        activ_fn = torch.nn.SELU()
    elif var == 'gelu':
        activ_fn = torch.nn.GELU()
    elif var == 'silu':
        activ_fn = torch.nn.SiLU()
    elif var == 'softsign':
        activ_fn = torch.nn.Softsign()
    elif var == 'sigmoid':
        activ_fn = torch.nn.Sigmoid()
    elif var == 'hardsigmoid':
        activ_fn = torch.nn.Hardsigmoid()
    else:
        raise ValueError(f'Unknown activation function {var}')
    return activ_fn


class Projection(Module):
    """
    An output projecton module.
    """

    def __init__(self, output_dim: 'int', input_dim: 'int', config: 'ProjectionConfig', device: 'Optional[Device]'=None, dtype: 'Optional[DataType]'=None) ->None:
        super().__init__()
        self.dtype = dtype
        init_fn = get_init_fn(config.linear_init_fn)
        lin = Linear(input_dim, output_dim, bias=config.linear_bias, device=device, dtype=dtype, init_fn=init_fn)
        if config.weight_normalization:
            self.fc = torch.nn.utils.parametrizations.weight_norm(lin)
        else:
            self.fc = lin
        self.activation_fn = parse_activation_fn(config.activation_name)
        if self.activation_fn is not None:
            self.activation_fn

    def forward(self, seqs: 'Tensor'):
        seqs = self.fc(seqs)
        if self.activation_fn is not None:
            seqs = self.activation_fn(seqs)
        return seqs


class DiTTimestepEncoder(Module):
    """
    Embeds scalar timesteps into vector representations.
    Based on DiT's `TimestepEmbedder`
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """

    def __init__(self, embedding_dim: 'int', frequency_embedding_size: 'int'=256, activation_fn_name: 'str'='silu', device: 'Optional[Device]'=None, dtype: 'Optional[DataType]'=None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.embedding_dim = embedding_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = Linear(frequency_embedding_size, embedding_dim, bias=True, device=device, dtype=dtype)
        self.nonlin = parse_activation_fn(activation_fn_name)
        self.fc2 = Linear(embedding_dim, embedding_dim, bias=True, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        """Reset the parameters and buffers of the module."""
        torch.nn.init.normal_(self.fc1.weight, std=0.02)
        torch.nn.init.normal_(self.fc2.weight, std=0.02)
        if self.fc1.bias is not None:
            torch.nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            torch.nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def sinusoidal_timestep_embedding(timestep, frequency_embedding_size, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timestep: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param frequency_embedding_size: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.

        Based on DiT's `TimestepEmbedder`
        https://github.com/facebookresearch/DiT/blob/main/models.py
        """
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: 'Tensor') ->Tensor:
        initial_size = timesteps.size()
        flat_timesteps = timesteps.view(-1, 1)
        t_freq = self.sinusoidal_timestep_embedding(flat_timesteps, self.frequency_embedding_size)
        t_emb = self.fc1(t_freq)
        if self.nonlin is not None:
            t_emb = self.nonlin(t_emb)
        t_emb = self.fc2(t_emb)
        return t_emb.view(*initial_size, self.embedding_dim)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaLNModulator,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_large_concept_model(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

