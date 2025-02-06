import sys
_module = sys.modules[__name__]
del sys
optimize_sd15_with_controlnet_and_ip_adapter = _module
reproduce_vae_segfault = _module
optimize_instant_id_pipeline = _module
optimize_lcm_lora = _module
optimize_lcm_pipeline = _module
optimize_stable_diffusion_pipeline = _module
optimize_stable_video_diffusion_pipeline = _module
optimize_train_text_to_image_lora = _module
setup = _module
sfast = _module
compilers = _module
diffusion_pipeline_compiler = _module
stable_diffusion_pipeline_compiler = _module
cuda = _module
graphs = _module
dynamo = _module
backends = _module
registry = _module
sfast_jit = _module
hooks = _module
module_jit_hook = _module
jit = _module
overrides = _module
passes = _module
triton_passes = _module
trace_helper = _module
utils = _module
libs = _module
diffusers = _module
image_processor = _module
xformers_attention = _module
xformers = _module
xformers_attention = _module
profile = _module
auto_profiler = _module
cprofile = _module
pretty_profile = _module
triton = _module
modules = _module
diffusers = _module
native = _module
patch = _module
ops = _module
activation = _module
conv = _module
copy = _module
group_norm = _module
layer_norm = _module
torch_ops = _module
aot_printer = _module
compute_precision = _module
copy = _module
copy_func = _module
custom_python_operator = _module
env = _module
flat_tensors = _module
gpu_device = _module
memory_format = _module
term_image = _module
climage = _module
image_to_ansi = _module
imgcat = _module
kdtree = _module
torch_dispatch = _module
tests = _module
test_stable_diffusion_pipeline_compiler = _module
conftest = _module
test_graphs = _module
test_trace_helper = _module
operators = _module
test_cudnn_convolution = _module
test_cutlass_dual_linear = _module
test_cutlass_qlinear = _module
test_torch_ops = _module
test_torch_dispatch = _module

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


import numpy as np


import torch.nn.functional as F


import inspect


import time


import logging


import math


import random


import torch.utils.checkpoint


from torchvision import transforms


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CUDNN_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import functools


from torch._dynamo.backends.registry import register_backend


from torch._dynamo.backends.common import aot_autograd


from torch._dynamo.backends.common import fake_tensor_unsupported


from torch.overrides import TorchFunctionMode


from typing import List


from typing import Optional


import torch.nn as nn


from torch._prims_common import suggest_memory_format


from itertools import product


import itertools


from typing import Sequence


from torch.utils._python_dispatch import TorchDispatchMode


import copy


from torch.hub import download_url_to_file


from torch.hub import get_dir


class TracedPosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module, *, training=None):
        super().__init__()
        self.module = module
        if training is None:
            training = getattr(module, 'training', False) if isinstance(module, torch.nn.Module) else False
        self.train(training)

    def forward(self, *args, **kwargs):
        outputs = self.module(*self.convert_inputs(args, kwargs))
        unflat_outputs = flat_tensors.unflattern(outputs)
        return unflat_outputs

    def convert_inputs(self, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return flat_tensors.flattern((args, kwargs))


class TraceablePosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        training = getattr(module, 'training', False) if isinstance(module, torch.nn.Module) else False
        self.train(training)

    def forward(self, *args):
        orig_args, orig_kwargs = flat_tensors.unflattern(args)
        outputs = self.module(*orig_args, **orig_kwargs)
        flat_outputs = flat_tensors.flattern(outputs)
        return flat_outputs


class TritonLoRACompatibleConv(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def set_lora_layer(self, lora_layer):
        if hasattr(self.module, 'set_lora_layer'):
            self.module.set_lora_layer(lora_layer)

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonLoRACompatibleLinear(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def set_lora_layer(self, lora_layer):
        if hasattr(self.module, 'set_lora_layer'):
            self.module.set_lora_layer(lora_layer)

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonConv2D(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonLinear(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonGroupNorm(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        module = self.module
        return TTO.group_norm(x, module.num_groups, module.weight, module.bias, module.eps)


class TritonGroupNormSiLU(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        module = self.module
        return TTO.group_norm_silu(x, module.num_groups, module.weight, module.bias, module.eps)


class ConvBiasAddActivation(torch.nn.Module):

    def __init__(self, bias=True, activation_cls=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 3, bias=bias)
        self.act = activation_cls() if activation_cls is not None else torch.nn.Identity()

    def forward(self, x, y=None, alpha=1.0):
        x = self.conv(x)
        if y is not None:
            x = x.add(y, alpha=alpha)
        x = self.act(x)
        return x


class FusedConvBiasAddActivation(torch.nn.Module):

    def __init__(self, m):
        super().__init__()
        self.conv = m.conv
        self.act = m.act
        self.train(m.training)

    def forward(self, x, y=None, alpha=1.0):
        raise NotImplementedError()


class GEGLU(nn.Module):
    """
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: 'int', dim_out: 'int', bias=True):
        super().__init__()
        linear_cls = nn.Linear
        self.proj = linear_cls(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: 'torch.Tensor') ->torch.Tensor:
        if gate.device.type != 'mps':
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32))

    def forward(self, hidden_states, enable_opt=False):
        if enable_opt:
            return torch.ops.sfast.cutlass_linear_geglu_unified(hidden_states, self.proj.weight, self.proj.bias)
        else:
            hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
            return hidden_states * self.gelu(gate)


class LinearModule(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBiasAddActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64])], {}),
     False),
    (GEGLU,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearModule,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_chengzeyi_stable_fast(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

