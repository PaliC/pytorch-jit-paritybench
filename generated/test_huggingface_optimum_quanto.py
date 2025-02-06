import sys
_module = sys.modules[__name__]
del sys
evaluate_configurations = _module
evaluate_model = _module
gen_barchart = _module
metrics = _module
latency = _module
perplexity = _module
prediction = _module
setup = _module
awq = _module
bnb = _module
hqq = _module
quanto = _module
benchmark = _module
benchmark_marlin_fp8 = _module
benchmark_w4a16 = _module
test_int_mm = _module
test_int_mm_inductor = _module
test_weight_int4pack_mm = _module
test_weight_int8pack_mm = _module
quantize_sst2_model = _module
quantize_causal_lm_model = _module
quantize_asr_model = _module
quantize_StableDiffusion = _module
quantize_mnist_model = _module
quantize_vit_model = _module
quantize_owl_model = _module
quantize_pixart_sigma = _module
conftest = _module
pack_intweight = _module
packing_utils = _module
test_awq_kernels = _module
test_awq_packing = _module
test_awq_quantize = _module
smoothquant = _module
calibrate = _module
library = _module
extensions = _module
cpp = _module
cuda = _module
extension = _module
hip = _module
mps = _module
qbytes_mm = _module
quantize = _module
unpack = _module
models = _module
diffusers_models = _module
shared_dict = _module
transformers_models = _module
nn = _module
qconv2d = _module
qlayernorm = _module
qlinear = _module
qmodule = _module
quantize = _module
subpackage = _module
commands = _module
base = _module
quantize = _module
tensor = _module
activations = _module
qbytes = _module
qbytes_ops = _module
quantization = _module
core = _module
function = _module
grouped = _module
optimizers = _module
absmax_optimizer = _module
affine_optimizer = _module
hqq_optimizer = _module
max_optimizer = _module
optimizer = _module
symmetric_optimizer = _module
packed = _module
qbits = _module
qbytes = _module
qtensor = _module
qtype = _module
weights = _module
packed = _module
qbits = _module
marlin = _module
fp8 = _module
packed = _module
qbits = _module
int4 = _module
packed = _module
qbits = _module
permutations = _module
packing = _module
qbits = _module
qbytes = _module
quantization = _module
reordering = _module
tinygemm = _module
packed = _module
qbits = _module
cli_helpers = _module
test_quantize_cli = _module
conftest = _module
helpers = _module
test_extensions = _module
test_mm = _module
test_quantize = _module
test_unpack = _module
test_quantized_model_for_causal_lm = _module
test_quantized_model_for_pixart = _module
test_calibrate = _module
test_qattention = _module
test_qconv2d = _module
test_qlayernorm = _module
test_qlinear = _module
test_qmodule = _module
test_quantize_mlp = _module
test_quantize_patterns = _module
test_requantize = _module
test_activations_compile = _module
test_activations_dispatch = _module
test_activations_quantize = _module
test_linear_dispatch = _module
test_mm_dispatch = _module
test_hqq_optimizer = _module
test_absmax = _module
test_packed_tensor = _module
test_awq_packed_tensor = _module
test_awq_weight_qbits_tensor = _module
test_marlin_fp8_packed_tensor = _module
test_marlin_int4_packed_tensor = _module
test_marlin_int4_weight_qbits_tensor = _module
test_marlin_qbytes_tensor = _module
test_tinygemm_packed_tensor = _module
test_tinygemm_weight_qbits_tensor = _module
test_weight_qbits_tensor = _module
test_weight_qbits_tensor_dispatch = _module
test_weight_qbits_tensor_instantiate = _module
test_weight_qbits_tensor_quantize = _module
test_weight_qbytes_tensor_backward = _module
test_weight_qbytes_tensor_dispatch = _module
test_weight_qbytes_tensor_instantiate = _module
test_weight_qbytes_tensor_quantize = _module
test_weight_qbytes_tensor_serialization = _module
weight_helpers = _module

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


import matplotlib.pyplot as plt


import numpy as np


import time


from typing import Optional


from functools import partial


import torch.utils.benchmark as benchmark


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


import functools


import torch.nn as nn


from torch.nn.modules.module import register_module_forward_hook


from torch.nn.modules.module import register_module_forward_pre_hook


from torch.overrides import TorchFunctionMode


import warnings


from typing import List


from torch.utils.cpp_extension import load


from typing import Union


from collections.abc import Mapping


from typing import Any


from typing import Dict


from abc import ABC


from typing import TYPE_CHECKING


from torch.autograd import Function


import numbers


from typing import Callable


import math


from typing import Tuple


from torch.utils import _pytree as pytree


from copy import copy


from enum import Enum


import uuid


import torch.utils.checkpoint


from torch import nn


class Optimizer(ABC):

    def __call__(self, base: 'torch.Tensor', bits: 'int', axis: 'int', group_size: 'Optional[int]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


class SymmetricOptimizer(Optimizer):

    def __call__(self, base: 'torch.Tensor', qtype: 'qtype', axis: 'Optional[int]'=None) ->torch.Tensor:
        if axis not in [None, 0, -1]:
            raise ValueError('axis parameter must be None, 0 (first axis) or -1 (last axis)')
        if axis is not None and base.shape[axis] == 1:
            axis = None
        scale = self.optimize(base, qtype, axis)
        assert scale.dtype == base.dtype
        return scale

    def optimize(self, base: 'torch.Tensor', qmax: 'float', axis: 'Optional[int]'=None) ->torch.Tensor:
        raise NotImplementedError


class AbsmaxOptimizer(SymmetricOptimizer):

    def optimize(self, base: 'torch.Tensor', qtype: 'qtype', axis: 'Optional[int]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        base = torch.abs(base)
        if axis is None:
            rmax = torch.max(base)
        else:
            dim = list(range(1, base.ndim)) if axis == 0 else list(range(0, base.ndim - 1))
            rmax = torch.amax(torch.abs(base), dim=dim, keepdim=True)
        return rmax / qtype.qmax

