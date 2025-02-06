import sys
_module = sys.modules[__name__]
del sys
_config = _module
boft_controlnet = _module
eval = _module
test_controlnet = _module
train_controlnet = _module
utils = _module
args_loader = _module
dataset = _module
light_controlnet = _module
pipeline_controlnet = _module
tracemalloc = _module
unet_2d_condition = _module
boft_dreambooth = _module
train_dreambooth = _module
args_loader = _module
dataset = _module
tracemalloc = _module
bone_finetuning = _module
peft_lora_clm_accelerate_ds_zero3_offload = _module
peft_adalora_seq2seq = _module
peft_lora_seq2seq_accelerate_ds_zero3_offload = _module
peft_lora_seq2seq_accelerate_fsdp = _module
corda_finetuning = _module
datautils = _module
preprocess = _module
dora_finetuning = _module
load_with_dora = _module
eva_finetuning = _module
eva_finetuning_multi_gpu = _module
utils = _module
peft_lora_embedding_semantic_search = _module
finetune_fp4_opt_bnb_peft = _module
train_dreambooth = _module
args_loader = _module
dataset = _module
tracemalloc = _module
fine_tune_blip2_int8 = _module
peft_adalora_whisper_large_training = _module
quantize_save_load = _module
train_gsm8k_llama = _module
convert_kohya_ss_sd_lora_to_peft = _module
convert_peft_sd_lora_to_kohya_ss = _module
train_dreambooth = _module
train_dreambooth = _module
olora_finetuning = _module
pissa_finetuning = _module
preprocess = _module
peft_no_lora_accelerate = _module
train = _module
utils = _module
convert_sd_adapter_to_peft = _module
train_dreambooth = _module
xlora_inference_mistralrs = _module
ci_clean_cache = _module
launch_notebook_mp = _module
log_reports = _module
stale = _module
setup = _module
peft = _module
auto = _module
config = _module
helpers = _module
import_utils = _module
mapping = _module
mapping_func = _module
mixed_model = _module
optimizers = _module
loraplus = _module
peft_model = _module
tuners = _module
_buffer_dict = _module
adalora = _module
bnb = _module
gptq = _module
layer = _module
model = _module
adaption_prompt = _module
layer = _module
model = _module
utils = _module
boft = _module
fbd = _module
layer = _module
model = _module
bone = _module
layer = _module
model = _module
cpt = _module
model = _module
fourierft = _module
layer = _module
model = _module
hra = _module
layer = _module
model = _module
ia3 = _module
bnb = _module
layer = _module
model = _module
ln_tuning = _module
layer = _module
model = _module
loha = _module
layer = _module
model = _module
lokr = _module
layer = _module
model = _module
lora = _module
aqlm = _module
awq = _module
bnb = _module
config = _module
corda = _module
dora = _module
eetq = _module
eva = _module
gptq = _module
hqq = _module
layer = _module
model = _module
torchao = _module
tp_layer = _module
lycoris_utils = _module
mixed = _module
model = _module
multitask_prompt_tuning = _module
model = _module
oft = _module
layer = _module
model = _module
p_tuning = _module
model = _module
poly = _module
layer = _module
model = _module
router = _module
prefix_tuning = _module
model = _module
prompt_tuning = _module
model = _module
tuners_utils = _module
vblora = _module
layer = _module
model = _module
vera = _module
bnb = _module
layer = _module
model = _module
xlora = _module
classifier = _module
layer = _module
model = _module
constants = _module
hotswap = _module
incremental_pca = _module
integrations = _module
loftq_utils = _module
merge_utils = _module
other = _module
peft_types = _module
save_and_load = _module
tests = _module
test_bnb_regression = _module
conftest = _module
test_regression = _module
run_compiled_diffusion_model_hotswap = _module
run_compiled_model_hotswap = _module
test_adaption_prompt = _module
test_auto = _module
test_boft = _module
test_common_gpu = _module
test_config = _module
test_cpt = _module
test_custom_models = _module
test_decoder_models = _module
test_encoder_decoder_models = _module
test_feature_extraction_models = _module
test_gptqmodel = _module
test_gpu_examples = _module
test_helpers = _module
test_hub_features = _module
test_incremental_pca = _module
test_initialization = _module
test_integrations = _module
test_lora_megatron = _module
test_loraplus = _module
test_low_level_api = _module
test_mapping = _module
test_mixed = _module
test_multitask_prompt_tuning = _module
test_other = _module
test_poly = _module
test_stablediffusion = _module
test_torch_compile = _module
test_tuners_utils = _module
test_vblora = _module
test_vera = _module
test_vision_models = _module
test_xlora = _module
testing_common = _module
testing_utils = _module

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


import numpy as np


import torch


from torchvision.utils import save_image


import time


import torch.utils.checkpoint


import itertools


import logging


import math


import torch.nn.functional as F


from typing import Optional


import random


from torchvision import transforms


from typing import Dict


from typing import List


from typing import Tuple


from typing import Union


from torch import nn


from torch.nn import functional as F


from typing import Any


from typing import Callable


import warnings


from torch.utils.data import Dataset


from typing import Literal


from torch.utils.data import DataLoader


import copy


from typing import Sequence


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.distributed import DistributedSampler


import torch.nn as nn


from random import randint


import re


from collections import Counter


from torch.optim import AdamW


from enum import Enum


from functools import lru_cache


from typing import TYPE_CHECKING


from torch.optim import Optimizer


import collections


import inspect


from copy import deepcopy


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from collections import OrderedDict


from torch.nn import Module


from torch.autograd import Function


from itertools import chain


from torch.nn.modules import Module


from typing import Set


from typing import Type


from typing import Iterable


from collections import defaultdict


from collections.abc import Mapping


from functools import partial


from itertools import cycle


from torch import svd_lowrank


from functools import reduce


import torch.nn.init as init


from abc import abstractmethod


from abc import ABC


from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


from torch.nn.init import _calculate_correct_fan


from torch import Tensor


import functools


from torch.testing import assert_close


from torch.distributed import init_process_group


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from scipy import stats


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(self, conditioning_embedding_channels: 'int', conditioning_channels: 'int'=3, block_out_channels: 'Tuple[int]'=(16, 32, 96, 256)):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
        self.conv_out = zero_module(nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1))

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class AutoModelForSentenceEmbedding(nn.Module):

    def __init__(self, model_name, tokenizer, normalize=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-09)

    def __getattr__(self, name: 'str'):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'model':
                raise
            return getattr(self.model, name)


class CastOutputToFloat(nn.Sequential):

    def forward(self, x):
        return super().forward(x)


class Shell(nn.Module):

    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


DUMMY_MODEL_CONFIG = {'model_type': 'custom'}


class BufferDict(Module):
    """
    Holds buffers in a dictionary.

    BufferDict can be indexed like a regular Python dictionary, but buffers it contains are properly registered, and
    will be visible by all Module methods. `torch.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and
    * in `torch.nn.BufferDict.update`, the order of the merged `OrderedDict` or another `torch.nn.BufferDict` (the
      argument to `torch.nn.BufferDict.update`).

    Note that `torch.nn.BufferDict.update` with other unordered mapping types (e.g., Python's plain `dict`) does not
    preserve the order of the merged mapping.

    Args:
        buffers (iterable, optional):
            a mapping (dictionary) of (string : `torch.Tensor`) or an iterable of key-value pairs of type (string,
            `torch.Tensor`)

    ```python
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.buffers = nn.BufferDict({"left": torch.randn(5, 10), "right": torch.randn(5, 10)})

        def forward(self, x, choice):
            x = self.buffers[choice].mm(x)
            return x
    ```
    """

    def __init__(self, buffers=None, persistent: 'bool'=False):
        """
        Args:
            buffers (`dict`):
                A mapping (dictionary) from string to `torch.Tensor`, or an iterable of key-value pairs of type
                (string, `torch.Tensor`).
        """
        super().__init__()
        if buffers is not None:
            self.update(buffers)
        self.persistent = persistent

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer, persistent=self.persistent)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key):
        """Remove key from the BufferDict and return its buffer.

        Args:
            key (`str`):
                Key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        """Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self):
        """Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self):
        """Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers):
        """
        Update the `torch.nn.BufferDict` with the key-value pairs from a mapping or an iterable, overwriting existing
        keys.

        Note:
            If `buffers` is an `OrderedDict`, a `torch.nn.BufferDict`, or an iterable of key-value pairs, the order of
            new elements in it is preserved.

        Args:
            buffers (iterable):
                a mapping (dictionary) from string to `torch.Tensor`, or an iterable of key-value pairs of type
                (string, `torch.Tensor`).
        """
        if not isinstance(buffers, collections.abc.Iterable):
            raise TypeError('BuffersDict.update should be called with an iterable of key/value pairs, but got ' + type(buffers).__name__)
        if isinstance(buffers, collections.abc.Mapping):
            if isinstance(buffers, (OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, collections.abc.Iterable):
                    raise TypeError('BufferDict update sequence element #' + str(j) + ' should be Iterable; is' + type(p).__name__)
                if not len(p) == 2:
                    raise ValueError('BufferDict update sequence element #' + str(j) + ' has length ' + str(len(p)) + '; 2 is required')
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else f' (GPU {p.get_device()})'
            parastr = f'Buffer containing: [{torch.typename(p)} of size {size_str}{device_str}]'
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('BufferDict should not be called.')


class BaseTunerLayer(ABC):
    """
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_pluggable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """
    adapter_layer_names: 'tuple[str, ...]' = ()
    other_param_names: 'tuple[str, ...]' = ()
    _disable_adapters: 'bool' = False
    _active_adapter: 'str | list[str]' = 'default'
    merged_adapters: 'list[str]' = []

    def get_base_layer(self) ->nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, 'base_layer'):
            base_layer = base_layer.base_layer
        return base_layer

    @property
    def weight(self) ->torch.Tensor:
        base_layer = self.get_base_layer()
        if hasattr(base_layer, 'qweight'):
            weight = base_layer.qweight
        else:
            weight = base_layer.weight
        return weight

    @property
    def bias(self) ->torch.Tensor:
        base_layer = self.get_base_layer()
        return base_layer.bias

    def merge(self, safe_merge: 'bool'=False, adapter_names: 'Optional[list[str]]'=None) ->None:
        raise NotImplementedError

    def unmerge(self) ->None:
        raise NotImplementedError

    @property
    def merged(self) ->bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) ->bool:
        return self._disable_adapters

    @property
    def active_adapter(self) ->(str | list[str]):
        return self._active_adapter

    def _get_available_adapters(self) ->set[str]:
        """Return all adapter names that can be found on this module."""
        adapters = set()
        for layer_name in self.adapter_layer_names:
            module = getattr(self, layer_name)
            if not isinstance(module, (nn.ModuleDict, nn.ParameterDict)):
                continue
            adapters.update(set(module.keys()))
        return adapters

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        return self.active_adapter

    def enable_adapters(self, enabled: 'bool') ->None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: 'str | list[str]') ->None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)
        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) ->list[str]:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in (self.adapter_layer_names + self.other_param_names):
            attr = getattr(self, name)
            if hasattr(attr, 'keys'):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: 'str') ->None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name (`str`): The name of the adapter to delete

        """
        for attr in (self.adapter_layer_names + self.other_param_names):
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]
        if adapter_name in self.active_adapters:
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(f'Adapter {adapter_name} was active which is now deleted. Setting active adapter to {new_active_adapter}.')
                    self.set_adapter(remaining_adapters[0])

    def _move_adapter_to_device_of_base_layer(self, adapter_name: 'str', device: 'Optional[torch.device]'=None) ->None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        if device is None:
            base_layer = self.get_base_layer()
            if isinstance(base_layer, nn.MultiheadAttention):
                base_layer = base_layer.out_proj
            for weight_name in ('weight', 'qweight'):
                weight = getattr(base_layer, weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                return
        meta = torch.device('meta')
        for adapter_layer_name in (self.adapter_layer_names + self.other_param_names):
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name]
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name]


DUMMY_TARGET_MODULES = 'dummy-target-modules'


EMBEDDING_LAYER_NAMES = ['embed_tokens', 'lm_head']


MIN_TARGET_MODULES_FOR_OPTIMIZATION = 20


class ModulesToSaveWrapper(torch.nn.Module):

    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = torch.nn.ModuleDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)
        self.check_module()

    def check_module(self):
        """Perform some sanity checks on the module to ensure that it works"""
        forbidden_classes = torch.nn.ModuleDict, torch.nn.ModuleList, torch.nn.ParameterDict, torch.nn.ParameterList
        if isinstance(self.original_module, forbidden_classes):
            cls_name = self.original_module.__class__
            raise TypeError(f'modules_to_save cannot be applied to modules of type {cls_name}')
        if isinstance(self.original_module, BaseTunerLayer):
            cls_name = self.original_module.__class__
            raise TypeError(f'modules_to_save cannot be applied to modules of type {cls_name}')

    @property
    def disable_adapters(self) ->bool:
        return self._disable_adapters

    @property
    def active_adapter(self) ->str:
        return self._active_adapter

    def __getattr__(self, name: 'str'):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if '_modules' not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        modules = self.__dict__['_modules']
        if self.disable_adapters:
            module = modules['original_module']
        elif self.active_adapter in modules['modules_to_save']:
            module = modules['modules_to_save'][self.active_adapter]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(module, name)

    def update(self, adapter_name):
        context_manager = nullcontext()
        for _, param in self.original_module.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, 'ds_numel'):
                context_manager = deepspeed.zero.GatheredParameters(self.original_module.parameters(), modifier_rank=0)
                break
        if adapter_name not in self.modules_to_save:
            with context_manager:
                self.modules_to_save[adapter_name] = copy.deepcopy(self.original_module)
        if hasattr(self.modules_to_save[adapter_name], '_hf_hook'):
            old_hook = self.modules_to_save[adapter_name]._hf_hook
            new_hook = self._create_new_hook(old_hook)
            remove_hook_from_module(self.modules_to_save[adapter_name])
            add_hook_to_module(self.modules_to_save[adapter_name], new_hook)
        self.original_module.requires_grad_(False)
        if adapter_name == self.active_adapter:
            self.modules_to_save[adapter_name].requires_grad_(True)

    def _create_new_hook(self, old_hook):
        """
        Creates a new hook based on the old hook. Use it only if you know what you are doing !
        """
        old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
        old_hook_attr = old_hook.__dict__
        filtered_old_hook_attr = {}
        old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
        for k in old_hook_attr.keys():
            if k in old_hook_init_signature.parameters:
                filtered_old_hook_attr[k] = old_hook_attr[k]
        new_hook = old_hook_cls(**filtered_old_hook_attr)
        return new_hook

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get('adapter_names', None)
        if adapter_names is None:
            return
        if len(x) != len(adapter_names):
            msg = f'Length of `adapter_names` should be the same as the number of inputs, but got {len(adapter_names)} and {len(x)} respectively.'
            raise ValueError(msg)

    def _mixed_batch_forward(self, input: 'torch.Tensor', *args: Any, adapter_names: list[str], **kwargs: Any) ->torch.Tensor:
        SUPPORTED_MODULES = torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d
        module_names = ', '.join([module.__name__ for module in SUPPORTED_MODULES])
        if not isinstance(self.original_module, SUPPORTED_MODULES):
            raise TypeError(f'Mixed batching is only supported for the following modules: {module_names}.')
        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
        results = [(0) for _ in range(len(input))]
        for i, active_adapter in enumerate(unique_adapters):
            sub_batch = input[sub_batch_indices_list[i]]
            if active_adapter == '__base__':
                output = self.original_module(sub_batch, *args, **kwargs)
            else:
                output = self.modules_to_save[active_adapter](sub_batch, *args, **kwargs)
            for index, j in enumerate(sub_batch_indices_list[i]):
                results[j] = output[index]
        return torch.stack(results)

    def forward(self, x: 'torch.Tensor', *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop('adapter_names', None)
        if self.disable_adapters or self.active_adapter not in self.modules_to_save:
            return self.original_module(x, *args, **kwargs)
        if adapter_names is None:
            return self.modules_to_save[self.active_adapter](x, *args, **kwargs)
        return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)

    def enable_adapters(self, enabled: 'bool'):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            return
        if enabled:
            self.original_module.requires_grad_(False)
            self.modules_to_save[self.active_adapter].requires_grad_(True)
            self._disable_adapters = False
        else:
            self.original_module.requires_grad_(True)
            self.modules_to_save.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_name: 'str'):
        """Set the active adapter

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.modules_to_save:
            raise ValueError(f'Adapter {adapter_name} not found in {self.modules_to_save.keys()}')
        self.modules_to_save[self.active_adapter].requires_grad_(False)
        self.modules_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name


CONFIG_NAME = 'adapter_config.json'


MIN_EXPECTED_CONFIG_KEYS = {'peft_type'}


class TaskType(str, enum.Enum):
    """
    Enum class for the different types of tasks supported by PEFT.

    Overview of the supported task types:
    - SEQ_CLS: Text classification.
    - SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
    - CAUSAL_LM: Causal language modeling.
    - TOKEN_CLS: Token classification.
    - QUESTION_ANS: Question answering.
    - FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features
      for downstream tasks.
    """
    SEQ_CLS = 'SEQ_CLS'
    SEQ_2_SEQ_LM = 'SEQ_2_SEQ_LM'
    CAUSAL_LM = 'CAUSAL_LM'
    TOKEN_CLS = 'TOKEN_CLS'
    QUESTION_ANS = 'QUESTION_ANS'
    FEATURE_EXTRACTION = 'FEATURE_EXTRACTION'


def _check_and_remove_unused_kwargs(cls, kwargs):
    """Make PEFT configs forward-compatible by removing unused kwargs that were added in later PEFT versions.

    This assumes that removing the unused kwargs will not affect the default behavior.

    Returns the filtered kwargs and the set of removed keys.
    """
    signature_parameters = inspect.signature(cls.__init__).parameters
    unexpected_kwargs = set(kwargs.keys()) - set(signature_parameters.keys())
    for key in unexpected_kwargs:
        del kwargs[key]
    return kwargs, unexpected_kwargs


class PeftType(str, enum.Enum):
    """
    Enum class for the different types of adapters in PEFT.

    Supported PEFT types:
    - PROMPT_TUNING
    - MULTITASK_PROMPT_TUNING
    - P_TUNING
    - PREFIX_TUNING
    - LORA
    - ADALORA
    - BOFT
    - ADAPTION_PROMPT
    - IA3
    - LOHA
    - LOKR
    - OFT
    - XLORA
    - POLY
    - LN_TUNING
    - VERA
    - FOURIERFT
    - HRA
    - BONE
    """
    PROMPT_TUNING = 'PROMPT_TUNING'
    MULTITASK_PROMPT_TUNING = 'MULTITASK_PROMPT_TUNING'
    P_TUNING = 'P_TUNING'
    PREFIX_TUNING = 'PREFIX_TUNING'
    LORA = 'LORA'
    ADALORA = 'ADALORA'
    BOFT = 'BOFT'
    ADAPTION_PROMPT = 'ADAPTION_PROMPT'
    IA3 = 'IA3'
    LOHA = 'LOHA'
    LOKR = 'LOKR'
    OFT = 'OFT'
    POLY = 'POLY'
    LN_TUNING = 'LN_TUNING'
    VERA = 'VERA'
    FOURIERFT = 'FOURIERFT'
    XLORA = 'XLORA'
    HRA = 'HRA'
    VBLORA = 'VBLORA'
    CPT = 'CPT'
    BONE = 'BONE'


class _ExcludedModule:
    """
    A private helper method used to represent excluded modules in the check_target_module_exists function.
    """

    def __bool__(self):
        return False


def _find_minimal_target_modules(target_modules: 'list[str] | set[str]', other_module_names: 'list[str] | set[str]') ->set[str]:
    """Find the minimal set of target modules that is sufficient to separate them from the other modules.

    Sometimes, a very large list of target_modules could be passed, which can slow down loading of adapters (e.g. when
    loaded from diffusers). It may be possible to condense this list from hundreds of items to just a handful of
    suffixes that are sufficient to distinguish the target modules from the other modules.

    Example:
        ```py
        >>> from peft.tuners.tuners_utils import _find_minimal_target_modules

        >>> target_modules = [f"model.decoder.layers.{i}.self_attn.q_proj" for i in range(100)]
        >>> target_modules += [f"model.decoder.layers.{i}.self_attn.v_proj" for i in range(100)]
        >>> other_module_names = [f"model.encoder.layers.{i}.self_attn.k_proj" for i in range(100)]
        >>> _find_minimal_target_modules(target_modules, other_module_names)
        {"q_proj", "v_proj"}
        ```

    Args:
        target_modules (`list[str]` | `set[str]`):
            The list of target modules.
        other_module_names (`list[str]` | `set[str]`):
            The list of other module names. They must not overlap with the target modules.

    Returns:
        `set[str]`:
            The minimal set of target modules that is sufficient to separate them from the other modules.

    Raises:
        ValueError:
            If `target_modules` is not a list or set of strings or if it contains an empty string. Also raises an error
            if `target_modules` and `other_module_names` contain common elements.
    """
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError('target_modules should be a list or set of strings.')
    target_modules = set(target_modules)
    if '' in target_modules:
        raise ValueError('target_modules should not contain an empty string.')
    other_module_names = set(other_module_names)
    if not target_modules.isdisjoint(other_module_names):
        msg = 'target_modules and other_module_names contain common elements, this should not happen, please open a GitHub issue at https://github.com/huggingface/peft/issues with the code to reproduce this issue'
        raise ValueError(msg)

    def generate_suffixes(s):
        parts = s.split('.')
        return ['.'.join(parts[i:]) for i in range(len(parts))][::-1]
    other_module_suffixes = {suffix for item in other_module_names for suffix in generate_suffixes(item)}
    target_modules_suffix_map = {item: generate_suffixes(item) for item in target_modules}
    required_suffixes = set()
    for item, suffixes in sorted(target_modules_suffix_map.items(), key=lambda tup: tup[1]):
        for suffix in suffixes:
            if suffix in required_suffixes or suffix in other_module_suffixes:
                continue
            if not any(item.endswith('.' + req_suffix) for req_suffix in required_suffixes):
                required_suffixes.add(suffix)
                break
    if not required_suffixes:
        return set(target_modules)
    return required_suffixes


def _get_submodules(model, key):
    parent = model.get_submodule('.'.join(key.split('.')[:-1]))
    target_name = key.split('.')[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


INCLUDE_LINEAR_LAYERS_SHORTHAND = 'all-linear'


SEQ_CLS_HEAD_NAMES = ['score', 'classifier']


logger = logging.getLogger(__name__)


COMPATIBLE_TUNER_TYPES = PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.ADALORA, PeftType.OFT


TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {'t5': ['q', 'v'], 'mt5': ['q', 'v'], 'bart': ['q_proj', 'v_proj'], 'gpt2': ['c_attn'], 'bloom': ['query_key_value'], 'blip-2': ['q', 'v', 'q_proj', 'v_proj'], 'opt': ['q_proj', 'v_proj'], 'gptj': ['q_proj', 'v_proj'], 'gpt_neox': ['query_key_value'], 'gpt_neo': ['q_proj', 'v_proj'], 'bert': ['query', 'value'], 'roberta': ['query', 'value'], 'xlm-roberta': ['query', 'value'], 'electra': ['query', 'value'], 'deberta-v2': ['query_proj', 'value_proj'], 'deberta': ['in_proj'], 'layoutlm': ['query', 'value'], 'llama': ['q_proj', 'v_proj'], 'chatglm': ['query_key_value'], 'gpt_bigcode': ['c_attn'], 'mpt': ['Wqkv'], 'RefinedWebModel': ['query_key_value'], 'RefinedWeb': ['query_key_value'], 'falcon': ['query_key_value'], 'btlm': ['c_proj', 'c_attn'], 'codegen': ['qkv_proj'], 'mistral': ['q_proj', 'v_proj'], 'mixtral': ['q_proj', 'v_proj'], 'stablelm': ['q_proj', 'v_proj'], 'phi': ['q_proj', 'v_proj', 'fc1', 'fc2'], 'gemma': ['q_proj', 'v_proj'], 'gemma2': ['q_proj', 'v_proj'], 'qwen2': ['q_proj', 'v_proj']}


def check_target_module_exists(config, key: 'str') ->(bool | re.Match[str] | None):
    """A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
        None if no match found
    """
    if hasattr(config, 'exclude_modules') and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            if re.fullmatch(config.exclude_modules, key):
                return _ExcludedModule()
        elif key in config.exclude_modules:
            return _ExcludedModule()
        elif any(key.endswith(f'.{exclude_key}') for exclude_key in config.exclude_modules):
            return _ExcludedModule()
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    elif key in config.target_modules:
        target_module_found = True
    else:
        target_module_found = any(key.endswith(f'.{target_key}') for target_key in config.target_modules)
        layer_indexes = getattr(config, 'layers_to_transform', None)
        layers_pattern = getattr(config, 'layers_pattern', None)
        is_using_layer_indexes = layer_indexes is not None and (len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True)
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match('.*\\.[^.]*\\.(\\d+)\\.', key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(f'.*\\.{pattern}\\.(\\d+)\\.', key)
                    if layer_index is not None:
                        break
            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes
    return target_module_found


@lru_cache
def is_auto_gptq_available():
    if importlib.util.find_spec('auto_gptq') is not None:
        AUTOGPTQ_MINIMUM_VERSION = packaging.version.parse('0.5.0')
        version_autogptq = packaging.version.parse(importlib_metadata.version('auto_gptq'))
        if AUTOGPTQ_MINIMUM_VERSION <= version_autogptq:
            return True
        else:
            raise ImportError(f'Found an incompatible version of auto-gptq. Found version {version_autogptq}, but only versions above {AUTOGPTQ_MINIMUM_VERSION} are supported')


SAFETENSORS_WEIGHTS_NAME = 'adapter_model.safetensors'


def starcoder_model_postprocess_past_key_value(past_key_values):
    result = []
    for k in past_key_values:
        k = k[:, :, 0]
        k = k.permute([1, 2, 0, 3])
        k = k.reshape(*k.shape[:-2], -1)
        result.append(k)
    return tuple(result)


TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = {'gpt_bigcode': starcoder_model_postprocess_past_key_value}


WEIGHTS_NAME = 'adapter_model.bin'


def dequantize_bnb_weight(weight: 'torch.nn.Parameter', state=None):
    """Helper function to dequantize 4bit or 8bit bnb weights.

    Since dequantization is not supported on CPU, the weight will be temporarily moved to CUDA if necessary.
    """
    device = weight.device
    is_cpu = device.type == torch.device('cpu').type
    if is_cpu and torch.cuda.is_available():
        weight = weight
    cls_name = weight.__class__.__name__
    if cls_name == 'Params4bit':
        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        if is_cpu:
            dequantized = dequantized
        return dequantized
    if state.SCB is None:
        state.SCB = weight.SCB
    if hasattr(bnb.functional, 'int8_vectorwise_dequant'):
        dequantized = bnb.functional.int8_vectorwise_dequant(weight.data, state.SCB)
    else:
        dequantized = weight.data * state.SCB.view(-1, 1) * 0.007874015718698502
    if is_cpu:
        dequantized = dequantized
    return dequantized


def dequantize_module_weight(module: 'torch.nn.Module') ->torch.nn.Parameter:
    """
    Helper function to dequantize a quantized weight.

    This function should be extended if more quantization schemes are added to the library.

    If the weight is not quantized, it will be returned as is.
    """
    if hasattr(module, 'W_q'):
        weight = module.dequantize()
        return weight
    elif type(module.weight).__module__.startswith('torchao.'):
        weight = module.weight.dequantize()
        return weight
    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        if isinstance(weight, torch.Tensor):
            return weight
        raise TypeError(f'Input weight should be of type nn.Parameter, got {type(weight)} instead')
    cls_name = weight.__class__.__name__
    if cls_name not in ('Params4bit', 'Int8Params'):
        return weight
    quant_state = getattr(module, 'state', None)
    device = weight.device
    is_cpu = device.type == torch.device('cpu').type
    weight = dequantize_bnb_weight(weight, state=quant_state)
    if is_cpu:
        module.weight = module.weight
    return weight


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight
    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


class DoraLinearLayer(nn.Module):

    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    def get_weight_norm(self, weight, lora_weight, scaling) ->torch.Tensor:
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) ->None:
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()
        with gather_params_ctx(base_layer.parameters()):
            if base_layer.__class__.__name__ == 'Linear4bit':
                base_layer = deepcopy(base_layer)
            weight = dequantize_module_weight(base_layer)
            if weight.data.ndim >= 4:
                lora_weight = torch.mm(lora_B.flatten(start_dim=1), lora_A.flatten(start_dim=1))
                lora_weight = lora_weight.reshape(weight.shape)
            else:
                lora_weight = lora_B @ lora_A
            if dtype_is_fp16:
                lora_weight = lora_weight.half()
            weight_norm = self.get_weight_norm(weight, lora_weight, scaling)
        if place_on_cpu:
            weight_norm = weight_norm
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=x.dtype)
        lora_weight = lora_B(lora_A(x_eye)).T
        magnitude = self.weight
        weight = dequantize_module_weight(base_layer)
        weight = weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        lora_result = lora_B(lora_A(x))
        bias = None
        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                base_result = base_result - bias
        else:
            base_result = F.linear(x, transpose(weight, self.fan_in_fan_out))
        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling
        return result_dora

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.dora.' + rep


def get_bnb_param_type(param: 'torch.nn.Parameter') ->Literal[False, '4bit', '8bit']:
    """Returns '4bit' or '8bit' if bitsandbytes parameter, else False"""
    if param.__class__.__name__ == 'Params4bit':
        return '4bit'
    if param.__class__.__name__ == 'Int8Params':
        return '8bit'
    return False


@lru_cache
def is_xpu_available(check_device=False):
    """
    Checks if XPU acceleration is available and potentially if a XPU is in the environment
    """
    system = platform.system()
    if system == 'Darwin':
        return False
    else:
        if check_device:
            try:
                _ = torch.xpu.device_count()
                return torch.xpu.is_available()
            except RuntimeError:
                return False
        return hasattr(torch, 'xpu') and torch.xpu.is_available()


class LoraLayer(BaseTunerLayer):
    adapter_layer_names = 'lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B'
    other_param_names = 'r', 'lora_alpha', 'scaling', 'lora_dropout'

    def __init__(self, base_layer: 'nn.Module', ephemeral_gpu_offload: 'bool'=False, **kwargs) ->None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: 'dict[str, bool]' = {}
        self.lora_bias: 'dict[str, bool]' = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()
        self._caches: 'dict[str, Any]' = {}
        self.ephemeral_gpu_offload: 'bool' = ephemeral_gpu_offload
        self.kwargs = kwargs
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = base_layer.weight.ds_shape if hasattr(base_layer.weight, 'ds_shape') else base_layer.weight.shape
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f'Only same dim for query/key/value is supported as of now for {self.__class__}.')
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, 'infeatures') and hasattr(base_layer, 'outfeatures'):
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, 'input_size') and hasattr(base_layer, 'output_size'):
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, 'codebooks') and base_layer.__class__.__name__ == 'QuantizedLinear':
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, 'w_bit') and base_layer.__class__.__name__ == 'WQLinear_GEMM':
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == 'EetqLinear':
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, 'W_q') and base_layer.__class__.__name__ == 'HQQLinear':
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            if hasattr(base_layer, 'in_features') and hasattr(base_layer, 'out_features'):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning)
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: 'bool'=False, lora_bias: 'bool'=False):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith('pissa'):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith('corda'):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == 'olora':
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == 'loftq':
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == 'eva':
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == 'gaussian':
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f'Unknown initialization init_lora_weights={init_lora_weights!r}')
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            if self.lora_bias[adapter_name]:
                nn.init.zeros_(self.lora_B[adapter_name].bias)
        if adapter_name in self.lora_embedding_A.keys():
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def olora_init(self, adapter_name):
        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight
        bnb_param_type = get_bnb_param_type(orig_weight)
        dtype = orig_weight.dtype
        if bnb_param_type:
            weight_tensor = dequantize_module_weight(base_layer)
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = orig_weight
        else:
            raise TypeError(f'Unsupported data type for the base layer. Got {dtype}.')
        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor
        Q, R = torch.linalg.qr(weight_tensor.data)
        Qr, Rr = Q[:, :r], R[:r]
        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()
        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        if bnb_param_type == '4bit':
            weight_tensor = orig_weight.__class__(weight_tensor, quant_type=orig_weight.quant_type, quant_storage=orig_weight.quant_storage, compress_statistics=orig_weight.compress_statistics, module=orig_weight.module)
            base_layer.weight = weight_tensor
        elif bnb_param_type == '8bit':
            weight_tensor = orig_weight.__class__(weight_tensor, requires_grad=orig_weight.requires_grad, has_fp16_weights=orig_weight.has_fp16_weights)
            base_layer.weight = weight_tensor
        else:
            weight_tensor = weight_tensor
            base_layer.weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError('Please initialize PiSSA under float32, float16, or bfloat16. Subsequently, re-quantize the residual model to help minimize quantization errors.')
        weight = transpose(weight, self.fan_in_fan_out)
        if init_lora_weights == 'pissa':
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, :self.r[adapter_name]]
            Sr = S[:self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[:self.r[adapter_name]]
        elif len(init_lora_weights.split('_niter_')) == 2:
            Vr, Sr, Ur = svd_lowrank(weight.data, self.r[adapter_name], niter=int(init_lora_weights.split('_niter_')[-1]))
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead.")
        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = transpose(weight, self.fan_in_fan_out)
        self.get_base_layer().weight.data = weight

    def corda_init(self, adapter_name, init_lora_weights):
        linear = self.get_base_layer()
        weight = linear.weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError('Please initialize CorDA under float32, float16, or bfloat16. Subsequently, re-quantize the residual model to help minimize quantization errors.')
        weight = weight
        out_dim = weight.data.size(0)
        in_dim = weight.data.size(1)
        if not hasattr(linear, 'eigens'):
            raise ValueError('`eigens` attribute not found for layer, please run `preprocess_corda` first. More information can be found at examples/corda_finetuning/README.md.')
        eigens = linear.eigens
        U = eigens.U_WC
        S = eigens.S_WC
        V = eigens.V_WC
        r = self.r[adapter_name]
        if torch.isnan(S).any() or torch.isinf(S).any():
            raise ValueError('Invalid value found in matrix S. Please file an issue at https://github.com/huggingface/peft/issues.')
        if torch.isnan(U).any() or torch.isinf(U).any():
            raise ValueError('Invalid value found in matrix U. Please file an issue at https://github.com/huggingface/peft/issues.')
        if torch.isnan(V).any() or torch.isinf(V).any():
            raise ValueError('Invalid value found in matrix V. Please file an issue at https://github.com/huggingface/peft/issues.')
        if U.size(0) != out_dim or U.size(1) != r:
            raise ValueError(f"Matrix U size mismatch: {U.size()} vs. ({out_dim}, {r}). Please make sure the `lora_config` and `model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank.")
        if S.size(0) != r:
            raise ValueError(f"Matrix S size mismatch: {S.size()} vs. ({r},). Please make sure the `lora_config` and `model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank.")
        if V.size(0) != in_dim or V.size(1) != r:
            raise ValueError(f"Matrix V size mismatch: {V.size()} vs. ({in_dim}, {r}). Please make sure the `lora_config` and `model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank.")
        S /= self.scaling[adapter_name]
        lora_A = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        lora_B = U.mul(S.sqrt()).contiguous()
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight
        self.get_base_layer().weight.data = weight
        del linear.eigens

    def loftq_init(self, adapter_name):
        weight = self.get_base_layer().weight
        kwargs = {'num_bits': self.kwargs.get('loftq_bits', 4), 'reduced_rank': self.r[adapter_name], 'num_iter': self.kwargs.get('loftq_iter', 1)}
        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def dora_init(self, adapter_name: 'str') ->None:
        if not self.lora_magnitude_vector:
            self.adapter_layer_names = self.adapter_layer_names[:] + ('lora_magnitude_vector',)
        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, 'fan_in_fan_out', False))
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == 'cpu' or lora_B.device.type == 'cpu')
        if self.ephemeral_gpu_offload:
            if lora_A.device.type in ['cuda', 'xpu']:
                lora_B = lora_B
            else:
                if lora_B.device.type not in ['cuda', 'xpu']:
                    if is_xpu_available():
                        lora_B = lora_B
                    else:
                        lora_B = lora_B
                lora_A = lora_A
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _cache_store(self, key: 'str', value: 'Any') ->None:
        self._caches[key] = value

    def _cache_pop(self, key: 'str') ->Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: 'float') ->None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) ->None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get('adapter_names', None)
        if adapter_names is None:
            return
        if len(x) != len(adapter_names):
            msg = f'Length of `adapter_names` should be the same as the number of inputs, but got {len(adapter_names)} and {len(x)} respectively.'
            raise ValueError(msg)
        if self.merged:
            msg = 'Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first.'
            raise ValueError(msg)
        unique_adapters = {name for name in adapter_names if name != '__base__'}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = 'Cannot pass `adapter_names` when DoRA is enabled.'
                raise ValueError(msg)

    def _mixed_batch_forward(self, x: 'torch.Tensor', *args: Any, adapter_names: list[str], **kwargs: Any) ->torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == '__base__':
                continue
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            sub_batch = x[sub_batch_indices_list[i]]
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output
        return result


class AdaLoraLayer(LoraLayer):
    adapter_layer_names = 'lora_A', 'lora_B', 'lora_E', 'lora_embedding_A', 'lora_embedding_B'
    other_param_names = 'r', 'lora_alpha', 'scaling', 'lora_dropout', 'ranknum'

    def __init__(self, base_layer: 'nn.Module') ->None:
        super().__init__(base_layer)
        self.lora_E = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r < 0:
            raise ValueError(f'`r` should be a positive integer or 0, but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer
        self.lora_A[adapter_name] = nn.Parameter(torch.randn(r, self.in_features))
        self.lora_E[adapter_name] = nn.Parameter(torch.randn(r, 1))
        self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r))
        self.ranknum[adapter_name] = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.zeros_(self.lora_E[adapter_name])
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for OFT.
    """

    def __init__(self, p=0.0):
        """
        Initializes the multiplicative dropout layer.

        Parameters:
        p (float): The probability of dropping out a block. Defaults to 0.0.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Applies multiplicative dropout to the input tensor.

        Parameters:
        x (Tensor): The input tensor of shape (D, H, H), where `D` represents
                    the number of OFT blocks, and `H` is the size of the square blocks along the last two dimensions,
                    the block size in OFT.
        """
        if self.training:
            if x.shape[-1] != x.shape[-2]:
                raise ValueError('The last two dimensions of input should be the same!')
            D, H, _ = x.shape
            if D == 1:
                return x
            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])
            mask = mask[torch.randperm(D)].view(D, 1, 1)
            eye_matrix = torch.eye(H, device=x.device).repeat(D, 1, 1)
            x = (1 - mask) * x + mask * eye_matrix
        return x


class OFTLayer(BaseTunerLayer):
    """
    Implements the OFT layer.
    """
    adapter_layer_names = 'oft_r', 'oft_s'
    other_param_names = 'r', 'oft_block_size', 'oft_dropout'

    def __init__(self, base_layer: 'nn.Module', **kwargs) ->None:
        """
        Initializes the OFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be
        added soon.

        Parameters:
        base_layer: the pretrained model layer
        """
        self.base_layer = base_layer
        self.oft_r = nn.ParameterDict({})
        self.oft_s = nn.ParameterDict({})
        self.r = {}
        self.oft_block_size = {}
        self.oft_dropout = nn.ModuleDict({})
        self.coft = {}
        self.eps = {}
        self.block_share = {}
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        else:
            raise ValueError(f'Unsupported layer type {type(base_layer)}')
        self.in_features = in_features
        self.out_features = out_features

    @property
    def _available_adapters(self) ->set[str]:
        return {*self.oft_r}

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            return
        warnings.warn('Scaling operation for OFT not supported! Automatically set scale to 1.')

    def scale_layer(self, scale: 'float') ->None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_r.keys():
                continue
            warnings.warn('Scaling operation for OFT not supported! Automatically set scale to 1.')

    def unscale_layer(self, scale=None) ->None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_r.keys():
                continue
            warnings.warn('Unscaling operation for OFT not supported! Keeping scale to 1.')

    def update_layer(self, adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights):
        """
        Update the linear layer with trainable OFT weights. Override for other layer types.
        """
        """Internal function to create oft adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            oft_block_size (`int`): The block size for added adapter.
            module_dropout (`float`):
                The multiplicative dropout probability for disabling adapter blocks during training.
            coft (`bool`): Whether to use the constrained variant of OFT or not.
            eps (`float`):
                The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
            block_share (`bool`): Whether to share the OFT parameters between blocks or not.
            init_weights (`bool`): Whether to initialize weights.
        """
        if module_dropout > 0.0:
            oft_dropout_layer = MultiplicativeDropoutLayer(p=module_dropout)
        else:
            oft_dropout_layer = nn.Identity()
        self.oft_dropout.update(nn.ModuleDict({adapter_name: oft_dropout_layer}))
        if r == 0 and oft_block_size != 0:
            if self.in_features % oft_block_size != 0 or oft_block_size > self.in_features:
                old_oft_block_size = oft_block_size
                oft_block_size = self.adjust_oft_parameters(self.in_features, oft_block_size)
                warnings.warn(f'Invalid `oft_block_size` ({old_oft_block_size})! Adjusted `oft_block_size` to ({oft_block_size}).')
            r = int(self.in_features // oft_block_size)
        elif r != 0 and oft_block_size == 0:
            if self.in_features % r != 0 or r > self.in_features:
                old_r = r
                r = self.adjust_oft_parameters(self.in_features, r)
                warnings.warn(f'Invalid `r` ({old_r})! Adjusted `r` to ({r}).')
            oft_block_size = int(self.in_features // r)
        else:
            raise ValueError('Something went wrong, please report this error: https://github.com/huggingface/peft/issues')
        self.coft[adapter_name] = coft
        self.block_share[adapter_name] = block_share
        self.eps[adapter_name] = eps * math.ceil(self.out_features / r) * math.ceil(self.out_features / r)
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(1, math.ceil(self.in_features / r), math.ceil(self.in_features / r)))
        else:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(r, math.ceil(self.in_features / r), math.ceil(self.in_features / r)))
        self.oft_s[adapter_name] = nn.Parameter(torch.empty(int(self.out_features), 1))
        self.reset_oft_parameters(adapter_name, init_weights)
        self.r[adapter_name] = r
        self.oft_block_size[adapter_name] = oft_block_size
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_oft_parameters(self, adapter_name, init_weights):
        """
        Reset the OFT parameters.
        """
        if init_weights is False:
            nn.init.normal_(self.oft_r[adapter_name], mean=0.0, std=0.1)
            nn.init.normal_(self.oft_s[adapter_name], mean=1.0, std=0.1)
            return
        if adapter_name in self.oft_r.keys():
            if init_weights is True:
                nn.init.zeros_(self.oft_r[adapter_name])
                nn.init.ones_(self.oft_s[adapter_name])
            else:
                raise ValueError(f'Unknown initialization init_weights={init_weights!r}')

    def _cayley_batch(self, data: 'torch.Tensor') ->torch.Tensor:
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.

        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """
        b, r, c = data.shape
        skew_mat = 0.5 * (data - data.transpose(1, 2))
        id_mat = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)
        Q = torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat, left=False)
        return Q

    def _block_diagonal(self, oft_r: 'torch.Tensor', rank: 'int') ->torch.Tensor:
        if oft_r.shape[0] == 1:
            blocks = [oft_r[0, ...] for i in range(rank)]
        else:
            blocks = [oft_r[i, ...] for i in range(rank)]
        A = torch.block_diag(*blocks)
        return A

    def _project_batch(self, oft_r, eps=1e-05):
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0]))
        I = torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype).unsqueeze(0).expand_as(oft_r)
        diff = oft_r - I
        norm_diff = torch.norm(oft_r - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_r, I + eps * (diff / norm_diff))
        return out

    def adjust_oft_parameters(self, in_features, params):
        """
        Adjust the OFT parameters to be divisible by the in_features dimension.
        """
        if params < in_features:
            higher_params = params
            while higher_params <= in_features and in_features % higher_params != 0:
                higher_params += 1
        else:
            return in_features
        lower_params = params
        while lower_params > 1 and in_features % lower_params != 0:
            lower_params -= 1
        if params - lower_params <= higher_params - params:
            return lower_params
        else:
            return higher_params


def check_adapters_to_merge(module: 'BaseTunerLayer', adapter_names: 'Optional[list[str]]'=None) ->list[str]:
    """
    Helper function to check which adapters should be merged.

    Only return those adapters that are not already merged. Give a warning if some or all of the adapters are already
    merged.

    """
    if adapter_names is None:
        adapter_names = module.active_adapters
    if isinstance(adapter_names, str):
        raise ValueError(f'adapter_names should be a list of strings, got {adapter_names!r}.')
    if module.merged:
        merged_adapters = set(module.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]
        if adapter_names:
            warnings.warn(f"Already following adapters were merged {','.join(module.merged_adapters)}. You are now additionally merging {','.join(adapter_names)}.")
        else:
            warnings.warn('All adapters are already merged, nothing to do.')
    return adapter_names


class Conv2d(nn.Module, OFTLayer):
    """OFT implemented in Conv2d layer"""

    def __init__(self, base_layer: 'nn.Module', adapter_name: 'str', r: 'int'=8, oft_block_size: 'int'=0, fan_in_fan_out: 'bool'=False, module_dropout: 'float'=0.0, coft: 'bool'=False, eps: 'float'=6e-05, block_share: 'bool'=False, init_weights: 'Union[bool, str]'=True, **kwargs) ->None:
        super().__init__()
        OFTLayer.__init__(self, base_layer)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights)

    def update_layer(self, adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights):
        """
        Update the conv2d layer with trainable OFT weights.
        """
        if module_dropout > 0.0:
            oft_dropout_layer = MultiplicativeDropoutLayer(p=module_dropout)
        else:
            oft_dropout_layer = nn.Identity()
        self.oft_dropout.update(nn.ModuleDict({adapter_name: oft_dropout_layer}))
        base_layer = self.get_base_layer()
        conv_filter_dim = self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]
        if r == 0 and oft_block_size != 0:
            if conv_filter_dim % oft_block_size != 0 or oft_block_size > conv_filter_dim:
                old_oft_block_size = oft_block_size
                oft_block_size = self.adjust_oft_parameters(conv_filter_dim, oft_block_size)
                warnings.warn(f'Invalid `oft_block_size` ({old_oft_block_size})! Adjusted `oft_block_size` to ({oft_block_size}).')
            r = int(conv_filter_dim // oft_block_size)
        elif r != 0 and oft_block_size == 0:
            if conv_filter_dim % r != 0 or r > conv_filter_dim:
                old_r = r
                r = self.adjust_oft_parameters(conv_filter_dim, r)
                warnings.warn(f'Invalid `r` ({old_r})! Adjusted `r` to ({r}).')
            oft_block_size = int(conv_filter_dim // r)
        else:
            raise ValueError('Something went wrong, please report this error: https://github.com/huggingface/peft/issues')
        self.coft[adapter_name] = coft
        self.block_share[adapter_name] = block_share
        self.eps[adapter_name] = eps * math.ceil(self.out_features / r) * math.ceil(self.out_features / r)
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(1, math.ceil(conv_filter_dim / r), math.ceil(conv_filter_dim / r)))
        else:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(r, math.ceil(conv_filter_dim / r), math.ceil(conv_filter_dim / r)))
        self.oft_s[adapter_name] = nn.Parameter(torch.empty(int(self.out_features), 1))
        self.reset_oft_parameters(adapter_name, init_weights)
        self.r[adapter_name] = r
        self.oft_block_size[adapter_name] = oft_block_size
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: 'bool'=False, adapter_names: 'Optional[list[str]]'=None) ->None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.oft_r.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    oft_mat, oft_s = self.get_delta_weight(active_adapter)
                    orig_weights = orig_weights.view(self.out_features, self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0])
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights * oft_s
                    orig_weights = orig_weights.view(self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0])
                    base_layer.weight.data = orig_weights.contiguous()
                else:
                    oft_mat, oft_s = self.get_delta_weight(active_adapter)
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights.view(self.out_features, self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0])
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights * oft_s
                    orig_weights = orig_weights.view(self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0])
                    base_layer.weight.data = orig_weights.contiguous()
                self.merged_adapters.append(active_adapter)

    def unmerge(self) ->None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.oft_r.keys():
                oft_mat, oft_s = self.get_delta_weight(active_adapter)
                orig_weights = self.get_base_layer().weight.data.clone()
                orig_weights = orig_weights.view(self.out_features, self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0])
                orig_weights = torch.transpose(orig_weights, 0, 1)
                orig_weights = torch.mm(oft_mat.t(), orig_weights)
                orig_weights = torch.transpose(orig_weights, 0, 1)
                orig_weights = orig_weights * (1 / oft_s)
                orig_weights = orig_weights.view(self.out_features, self.in_features, self.get_base_layer().kernel_size[0], self.get_base_layer().kernel_size[0])
                self.get_base_layer().weight.data = orig_weights

    def get_delta_weight(self, adapter_name) ->tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        oft_r = self.oft_r[adapter_name]
        oft_s = self.oft_s[adapter_name]
        rank = self.r[adapter_name]
        coft = self.coft[adapter_name]
        eps = self.eps[adapter_name]
        if coft:
            with torch.no_grad():
                oft_r.copy_(self._project_batch(oft_r, eps=eps))
        orth_rotate = self._cayley_batch(oft_r)
        weight = self._block_diagonal(orth_rotate, rank)
        return weight, oft_s

    def forward(self, x: 'torch.Tensor', *args: Any, **kwargs: Any) ->torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            oft_rotation = torch.eye(self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0], device=x.device, dtype=previous_dtype)
            oft_scale = torch.ones((int(self.out_features), 1), device=x.device, dtype=previous_dtype)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.oft_r.keys():
                    continue
                oft_r = self.oft_r[active_adapter]
                oft_s = self.oft_s[active_adapter]
                dropout = self.oft_dropout[active_adapter]
                rank = self.r[active_adapter]
                coft = self.coft[active_adapter]
                eps = self.eps[active_adapter]
                if coft:
                    with torch.no_grad():
                        oft_r.copy_(self._project_batch(oft_r, eps=eps))
                orth_rotate = self._cayley_batch(oft_r)
                orth_rotate = dropout(orth_rotate)
                oft_mat = self._block_diagonal(orth_rotate, rank)
                oft_rotation = oft_mat @ oft_rotation
                oft_scale = oft_s * oft_scale
            x = x
            orig_weights = self.base_layer.weight.data
            orig_weights = orig_weights.view(self.out_features, self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0])
            orig_weights = torch.transpose(orig_weights, 0, 1)
            oft_rotation = oft_rotation
            orig_weights = orig_weights
            rotated_weight = torch.mm(oft_rotation, orig_weights)
            rotated_weight = torch.transpose(rotated_weight, 0, 1)
            scaled_rotated_weight = rotated_weight * oft_scale
            scaled_rotated_weight = scaled_rotated_weight.view(self.out_features, self.in_features, self.get_base_layer().kernel_size[0], self.get_base_layer().kernel_size[0])
            result = F.conv2d(input=x, weight=scaled_rotated_weight, bias=self.get_base_layer().bias, padding=self.get_base_layer().padding[0], stride=self.get_base_layer().stride[0])
        result = result
        return result

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'oft.' + rep


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    kwargs['adapter_names'] = adapter_names
    return args, kwargs


def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


def magnitude_based_pruning(tensor: 'torch.Tensor', density: 'float') ->torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction
    `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The tensor with the pruned weights.
    """
    mask = torch.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.numel())
    top_k = torch.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: 'torch.Tensor', density: 'float', rescale: 'bool') ->torch.Tensor:
    """
    Prune random values based on the specified fraction `density`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        torch.div(input=pruned_tensor, other=density)
    return pruned_tensor


def prune(tensor: 'torch.Tensor', density: 'float', method: "Literal['magnitude', 'random']", rescale: 'bool'=False) ->torch.Tensor:
    """
    Prune the values of task tensors based on the `method`.

    Args:
        tensor (`torch.Tensor`):The tensor to prune.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        method (`str`):The method to use to prune. Should be one of ["magnitude", "random"].
        rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.

    Returns:
        `torch.Tensor`: The pruned tensor.
    """
    if density >= 1:
        warnings.warn(f'The density {density} is greater than or equal to 1, no pruning will be performed.')
        return tensor
    elif density < 0:
        raise ValueError(f'Density should be >= 0, got {density}')
    if method == 'magnitude':
        return magnitude_based_pruning(tensor, density)
    elif method == 'random':
        return random_pruning(tensor, density, rescale=rescale)
    else:
        raise ValueError(f'Unknown method {method}')


def reshape_weight_task_tensors(task_tensors, weights):
    """
    Reshapes `weights` to match the shape of `task_tensors` by unsqeezing in the remaining dimenions.

    Args:
        task_tensors (`torch.Tensor`): The tensors that will be used to reshape `weights`.
        weights (`torch.Tensor`): The tensor to be reshaped.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


def dare_linear(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor', density: 'float') ->torch.Tensor:
    """
    Merge the task tensors using `dare linear`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='random', rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def calculate_majority_sign_mask(tensor: 'torch.Tensor', method: "Literal['total', 'frequency']"='total') ->torch.Tensor:
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
        tensor (`torch.Tensor`):The tensor to get the mask from.
        method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The majority sign mask.
    """
    sign = tensor.sign()
    if method == 'total':
        sign_magnitude = tensor.sum(dim=0)
    elif method == 'frequency':
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors: 'torch.Tensor', majority_sign_mask: 'torch.Tensor') ->torch.Tensor:
    """
    Merge the task tensors using disjoint merge.

    Args:
        task_tensors (`torch.Tensor`):The task tensors to merge.
        majority_sign_mask (`torch.Tensor`):The mask of the majority sign across the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)


def dare_ties(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor', density: 'float', majority_sign_method: "Literal['total', 'frequency']"='total') ->torch.Tensor:
    """
    Merge the task tensors using `dare ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='random', rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


class AqlmLoraLinear(torch.nn.Module, LoraLayer):

    def __init__(self, base_layer, adapter_name: 'str', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, init_lora_weights: 'bool'=True, use_rslora: 'bool'=False, use_dora: 'bool'=False, lora_bias: 'bool'=False, **kwargs):
        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA yet, please set it to False')
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora, lora_bias=lora_bias)

    def forward(self, x: 'torch.Tensor'):
        result = self.base_layer(x)
        if self.disable_adapters:
            return result
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x
            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output
            output = output * scaling
            result += output
        return result

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.' + rep


@lru_cache
def is_aqlm_available():
    return importlib.util.find_spec('aqlm') is not None


def dispatch_aqlm(target: 'torch.nn.Module', adapter_name: 'str', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_aqlm_available() and isinstance(target_base_layer, QuantizedLinear):
        new_module = AqlmLoraLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.codes
    return new_module


class AwqLoraLinear(torch.nn.Module, LoraLayer):

    def __init__(self, base_layer, adapter_name, r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, init_lora_weights: 'bool'=True, use_rslora: 'bool'=False, use_dora: 'bool'=False, lora_bias: 'bool'=False, **kwargs):
        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA yet, please set it to False')
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self.quant_linear_module = base_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora, lora_bias=lora_bias)

    def forward(self, x: 'torch.Tensor'):
        result = self.quant_linear_module(x)
        if self.disable_adapters:
            return result
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x
            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output
            output = output * scaling
            result = result + output
        return result

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.' + rep


@lru_cache
def is_auto_awq_available():
    return importlib.util.find_spec('awq') is not None


def dispatch_awq(target: 'torch.nn.Module', adapter_name: 'str', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_auto_awq_available():
        if isinstance(target_base_layer, WQLinear_GEMM):
            AUTOAWQ_MINIMUM_VERSION = packaging.version.parse('0.2.0')
            version_autoawq = packaging.version.parse(importlib_metadata.version('autoawq'))
            if AUTOAWQ_MINIMUM_VERSION > version_autoawq:
                raise ImportError(f'Found an incompatible version of auto-awq. Found version {version_autoawq}, but only versions above {AUTOAWQ_MINIMUM_VERSION} are supported for PEFT.')
            new_module = AwqLoraLinear(target, adapter_name, **kwargs)
            target.qweight = target_base_layer.qweight
    return new_module


class _DoraConvNdLayer(DoraLinearLayer):

    def get_weight_norm(self, weight, lora_weight, scaling) ->torch.Tensor:
        weight = weight + scaling * lora_weight
        dim = tuple(range(1, weight.dim()))
        weight_norm = weight.norm(p=2, dim=dim, keepdim=True).transpose(1, 0)
        return weight_norm

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        weight = base_layer.weight
        lora_weight = torch.mm(lora_B.weight.flatten(start_dim=1), lora_A.weight.flatten(start_dim=1))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = (mag_norm_scale - 1) * self.conv_fn(x, weight, bias=None, stride=base_layer.stride, padding=base_layer.padding, dilation=base_layer.dilation, groups=base_layer.groups) + mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_dora

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.dora.' + rep


class _ConvNd(nn.Module, LoraLayer):

    def __init__(self, base_layer: 'nn.Module', adapter_name: 'str', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, init_lora_weights: 'Union[bool, str]'=True, use_rslora: 'bool'=False, use_dora: 'bool'=False, lora_bias: 'bool'=False, **kwargs) ->None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora, lora_bias=lora_bias)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        conv_layer = type(base_layer)
        out_kernel = out_stride = (1,) * (self._kernel_dim - 2)
        self.lora_A[adapter_name] = conv_layer(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = conv_layer(r, self.out_features, out_kernel, out_stride, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights == 'loftq':
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False
        self.set_adapter(self.active_adapters)

    def _get_dora_factor_view(self):
        return (-1,) + (1,) * (self._kernel_dim - 1)

    def dora_init(self, adapter_name: 'str') ->None:
        if self.lora_magnitude_vector is None:
            self.adapter_layer_names = self.adapter_layer_names[:] + ('lora_magnitude_vector',)
        dora_layer_class = self._get_dora_layer_class()
        dora_layer = dora_layer_class(fan_in_fan_out=False)
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _get_dora_layer_class(self) ->type[_DoraConvNdLayer]:
        raise NotImplementedError

    def merge(self, safe_merge: 'bool'=False, adapter_names: 'Optional[list[str]]'=None) ->None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        weight_norm = self.lora_magnitude_vector[active_adapter].get_weight_norm(orig_weights, delta_weight, scaling=1).detach()
                        self._cache_store(f'{active_adapter}-weight_norm', weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = dora_factor.view(*self._get_dora_factor_view()) * (orig_weights + delta_weight)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    base_layer.weight.data = orig_weights
                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                        base_layer.bias.data = new_bias
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        weight_norm = self.lora_magnitude_vector[active_adapter].get_weight_norm(base_layer.weight, delta_weight, scaling=1).detach()
                        self._cache_store(f'{active_adapter}-weight_norm', weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(*self._get_dora_factor_view()) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight
                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias
                self.merged_adapters.append(active_adapter)

    def unmerge(self) ->None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f'{active_adapter}-weight_norm')
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(*self._get_dora_factor_view()) - delta_weight
                    weight.data = weight_orig
                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias

    def get_delta_weight(self, adapter) ->torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype
        cast_to_fp32 = device.type == 'cpu' and (dtype == torch.float16 or dtype == torch.bfloat16)
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * self.scaling[adapter]
        else:
            output_tensor = self.conv_fn(weight_A.transpose(0, 1), weight_B).transpose(0, 1) * self.scaling[adapter]
        if cast_to_fp32:
            output_tensor = output_tensor
            self.lora_A[adapter].weight.data = weight_A
            self.lora_B[adapter].weight.data = weight_B
        return output_tensor

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop('adapter_names', None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x
                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](x, lora_A=lora_A, lora_B=lora_B, scaling=scaling, base_layer=self.get_base_layer())
            result = result
        return result

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.' + rep


class Conv1d(_ConvNd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 3:
            raise ValueError(f'Conv1d layer kernel must have 3 dimensions, not {self._kernel_dim}')
        self.conv_fn = F.conv1d

    def _get_dora_layer_class(self):
        raise NotImplementedError


class DoraConv3dLayer(_DoraConvNdLayer):

    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv3d


class Conv3d(_ConvNd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f'Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}')
        self.conv_fn = F.conv3d

    def _get_dora_layer_class(self):
        return DoraConv3dLayer


class DoraEmbeddingLayer(DoraLinearLayer):

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, embed_fn):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        lora_weight = (lora_A @ lora_B).T
        magnitude = self.weight
        weight = base_layer.weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = mag_norm_scale * (embed_fn(x, lora_A) @ lora_B) * scaling
        return mag_norm_scale, result_dora

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.dora.' + rep


class Embedding(nn.Module, LoraLayer):

    def __init__(self, base_layer: 'nn.Module', adapter_name: 'str', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, init_lora_weights: 'Union[bool, str]'=True, use_rslora: 'bool'=False, use_dora: 'bool'=False, lora_bias: 'bool'=False, **kwargs) ->None:
        if lora_bias:
            raise ValueError(f'lora_bias={lora_bias} is not supported for {self.__class__.__name__}.')
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora, lora_bias=lora_bias)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        self.lora_bias[adapter_name] = lora_bias
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights == 'loftq':
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False
        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: 'str') ->None:
        if self.lora_magnitude_vector is None:
            self.adapter_layer_names = self.adapter_layer_names[:] + ('lora_magnitude_vector',)
        dora_layer = DoraEmbeddingLayer(fan_in_fan_out=True)
        lora_embedding_A = self.lora_embedding_A[adapter_name]
        lora_embedding_B = self.lora_embedding_B[adapter_name]
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_embedding_A, lora_B=lora_embedding_B, scaling=scaling)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def merge(self, safe_merge: 'bool'=False, adapter_names: 'Optional[list[str]]'=None) ->None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) ->None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) ->torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype
        cast_to_fp32 = device.type == 'cpu' and (dtype == torch.float16 or dtype == torch.bfloat16)
        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]
        if cast_to_fp32:
            output_tensor = output_tensor
            self.lora_embedding_A[adapter] = weight_A
            self.lora_embedding_B[adapter] = weight_B
        return output_tensor

    def _mixed_batch_forward(self, x: 'torch.Tensor', *args: Any, adapter_names: list[str], **kwargs: Any) ->torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == '__base__':
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue
            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += after_A @ embedding_B * scaling
        return result

    def _embed(self, input: 'torch.Tensor', weight: 'torch.Tensor') ->torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(input, weight, padding_idx=base_layer.padding_idx, max_norm=base_layer.max_norm, norm_type=base_layer.norm_type, scale_grad_by_freq=base_layer.scale_grad_by_freq, sparse=base_layer.sparse)

    def forward(self, x: 'torch.Tensor', *args: Any, **kwargs: Any) ->torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop('adapter_names', None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                if not self.use_dora[active_adapter]:
                    after_A = self._embed(x, embedding_A)
                    result = result + after_A @ embedding_B * scaling
                else:
                    mag_norm_scale, dora_result = self.lora_magnitude_vector[active_adapter](x, lora_A=embedding_A, lora_B=embedding_B, scaling=scaling, base_layer=self.get_base_layer(), embed_fn=self._embed)
                    result = mag_norm_scale * result + dora_result
            result = result
        return result

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.' + rep


class VeraLayer(BaseTunerLayer):
    adapter_layer_names = 'vera_lambda_b', 'vera_lambda_d'
    other_param_names = 'vera_A', 'vera_B'

    def __init__(self, base_layer: 'nn.Module', **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.vera_dropout = nn.ModuleDict({})
        self.vera_lambda_b = nn.ParameterDict({})
        self.vera_lambda_d = nn.ParameterDict({})
        self.vera_A: 'Optional[BufferDict]' = None
        self.vera_B: 'Optional[BufferDict]' = None
        self._disable_adapters = False
        self.merged_adapters = []
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = base_layer.weight.ds_shape if hasattr(base_layer.weight, 'ds_shape') else base_layer.weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) ->bool:
        return bool(self.merged_adapters)

    def update_layer(self, adapter_name, vera_A: 'BufferDict', vera_B: 'BufferDict', r, vera_dropout, init_weights, d_initial: 'float'=0.1):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        if vera_dropout > 0.0:
            vera_dropout_layer = nn.Dropout(p=vera_dropout)
        else:
            vera_dropout_layer = nn.Identity()
        self.vera_dropout.update(nn.ModuleDict({adapter_name: vera_dropout_layer}))
        self.vera_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self.vera_lambda_d[adapter_name] = nn.Parameter(torch.randn(r), requires_grad=True)
        self.vera_A = vera_A
        self.vera_B = vera_B
        if adapter_name not in vera_A:
            if len(self.vera_A) < 1:
                raise ValueError('The `vera_A` and `vera_B` buffers are empty. This should not happen. Please report this issue.')
            vera_A_param = list(self.vera_A.values())[0]
            vera_B_param = list(self.vera_B.values())[0]
            error_tmpl = '{} has a size of {} but {} or greater is required; this probably happened because an additional VeRA adapter was added after the first one with incompatible shapes.'
            if vera_A_param.shape[1] < self.in_features:
                raise ValueError(error_tmpl.format('vera_A', vera_A_param.shape[1], self.in_features))
            if vera_B_param.shape[0] < self.out_features:
                raise ValueError(error_tmpl.format('vera_B', vera_B_param.shape[0], self.out_features))
            error_tmpl = '{} has a size of {} but {} or greater is required; this probably happened because an additional VeRA adapter with a lower rank was added after the first one; loading the adapters in reverse order may solve this.'
            if vera_A_param.shape[0] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format('vera_A', vera_A_param.shape[0], self.r[adapter_name]))
            if vera_B_param.shape[1] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format('vera_B', vera_B_param.shape[1], self.r[adapter_name]))
            self.vera_A[adapter_name] = vera_A_param
            self.vera_B[adapter_name] = vera_B_param
        if init_weights:
            self.reset_vera_parameters(adapter_name, d_initial=d_initial)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_vera_parameters(self, adapter_name, d_initial: 'float'=0.1):
        if adapter_name in self.vera_lambda_d.keys():
            with torch.no_grad():
                nn.init.zeros_(self.vera_lambda_d[adapter_name]).fill_(d_initial)
                nn.init.zeros_(self.vera_lambda_b[adapter_name])


class Linear(nn.Linear, VeraLayer):

    def __init__(self, base_layer, vera_A: 'BufferDict', vera_B: 'BufferDict', adapter_name: 'str', r: 'int'=0, vera_dropout: 'float'=0.0, fan_in_fan_out: 'bool'=False, is_target_conv_1d_layer: 'bool'=False, init_weights: 'bool'=True, d_initial: 'float'=0.1, **kwargs) ->None:
        super(nn.Linear, self).__init__()
        VeraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, vera_A, vera_B, r, vera_dropout, init_weights, d_initial=d_initial)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: 'bool'=False, adapter_names: 'Optional[List[str]]'=None) ->None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.vera_lambda_d.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) ->None:
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.vera_lambda_d.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) ->torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        vera_A = self.vera_A[adapter]
        vera_B = self.vera_B[adapter]
        device = vera_B.device
        dtype = vera_B.dtype
        cast_to_fp32 = device.type == 'cpu' and (dtype == torch.float16 or dtype == torch.bfloat16)
        lambda_d = self.vera_lambda_d[adapter]
        lambda_b = self.vera_lambda_b[adapter]
        if cast_to_fp32:
            vera_A = vera_A.float()
            vera_B = vera_B.float()
            lambda_d = lambda_d.float()
            lambda_b = lambda_b.float()
        sliced_A = vera_A[:, :self.in_features]
        sliced_B = vera_B[:self.out_features, :]
        lambda_b = lambda_b.unsqueeze(-1)
        lambda_d = lambda_d.unsqueeze(-1)
        output_tensor = transpose(lambda_b * sliced_B @ (lambda_d * sliced_A), self.fan_in_fan_out)
        if cast_to_fp32:
            output_tensor = output_tensor
            self.vera_lambda_d[adapter].data = lambda_d
            self.vera_lambda_b[adapter].data = lambda_b
        return output_tensor

    def forward(self, x: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.vera_lambda_d.keys():
                    continue
                lambda_d = self.vera_lambda_d[active_adapter]
                lambda_b = self.vera_lambda_b[active_adapter]
                vera_A = self.vera_A[active_adapter]
                vera_B = self.vera_B[active_adapter]
                sliced_A = vera_A[:, :self.in_features]
                sliced_B = vera_B[:self.out_features, :]
                dropout = self.vera_dropout[active_adapter]
                x = x
                result = result + lambda_b * F.linear(lambda_d * F.linear(dropout(x), sliced_A), sliced_B)
        result = result
        return result

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'vera.' + rep


def skip_init_on_device(func):
    """
    Ignore the init_on_device context manager when calling the decorated function.

    This is a narrow use decorator that allows us to avoid initializing on meta device even when we're inside the
    init_empty_weights context.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _skip_init_on_device():
            return func(*args, **kwargs)
    return wrapper


class MultiheadAttention(nn.Module, LoraLayer):
    """LoRA implemented in a multihead attention layer

    This is currently only implemented for the case of `_qkv_same_embed_dim = True`, i.e. query, key, and value having
    the same dimension.

    Note: LoRA is applied to both the in_proj (query/key/value) and out_proj. There is currently no way to specify only
    one of them. Don't try to apply LoRA to the out_proj of MultiheadAttention by targeting that layer specifically,
    since the forward method of that layer is not being used, hence the LoRA adapter would be ignored.

    This is a little bit hacky because of the way that MultiheadAttention is implemented in PyTorch: There are no
    `nn.Linear` layers which we can hook onto or, in case of output projection, `.forward` is not used. This
    implementation works around these problems by merging the weights before the forward call and unmerging them after
    the forward call.
    """

    def __init__(self, base_layer, adapter_name: 'str', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, init_lora_weights: 'Union[bool, str]'=True, use_rslora: 'bool'=False, use_dora: 'bool'=False, **kwargs) ->None:
        if not getattr(base_layer, '_qkv_same_embed_dim', True):
            raise ValueError(f'Only same embed for query/key/value is supported as of now for {self.__class__.__name__}.')
        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA (yet), please set use_dora to False')
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        if isinstance(base_layer.out_proj, nn.Linear):
            self.base_layer.out_proj = Linear(base_layer.out_proj, adapter_name, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora, **kwargs)
        else:
            raise ValueError(f'out_proj must be an instance of nn.Linear for {self.__class__.__name__}.')
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    @property
    def embed_dim(self) ->int:
        return self.get_base_layer().embed_dim

    @property
    def kdim(self) ->Optional[int]:
        return self.get_base_layer().kdim

    @property
    def vdim(self) ->Optional[int]:
        return self.get_base_layer().vdim

    @property
    def _qkv_same_embed_dim(self) ->bool:
        return self.get_base_layer()._qkv_same_embed_dim

    @property
    def num_heads(self) ->int:
        return self.get_base_layer().num_heads

    @property
    def dropout(self) ->float:
        return self.get_base_layer().dropout

    @property
    def batch_first(self) ->bool:
        return self.get_base_layer().batch_first

    @property
    def head_dim(self) ->int:
        return self.get_base_layer().head_dim

    @property
    def in_proj_weight(self) ->nn.Parameter:
        return self.get_base_layer().in_proj_weight

    @property
    def in_proj_bias(self) ->nn.Parameter:
        return self.get_base_layer().in_proj_bias

    @property
    def out_proj(self) ->nn.Module:
        return self.get_base_layer().out_proj.get_base_layer()

    @property
    def bias_k(self) ->Optional[nn.Parameter]:
        return self.get_base_layer().bias_k

    @property
    def bias_v(self) ->Optional[nn.Parameter]:
        return self.get_base_layer().bias_v

    def merge_masks(self, *args, **kwargs) ->tuple[Optional[torch.Tensor], Optional[int]]:
        return self.get_base_layer().merge_masks(*args, **kwargs)

    @property
    def add_zero_attn(self) ->bool:
        return self.get_base_layer().add_zero_attn

    def update_layer(self, *args, **kwargs) ->None:
        super().update_layer(*args, **kwargs)
        self.base_layer.out_proj.update_layer(*args, **kwargs)

    def merge(self, safe_merge: 'bool'=False, adapter_names: 'Optional[list[str]]'=None) ->None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights_in = base_layer.in_proj_weight.data.detach().clone()
                    orig_weights_in += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights_in).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    orig_weights_out = base_layer.out_proj.weight.data.detach().clone()
                    orig_weights_out += base_layer.out_proj.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights_out).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    del base_layer.in_proj_weight
                    base_layer.in_proj_weight = orig_weights_in
                    del base_layer.out_proj.get_base_layer().weight
                    base_layer.out_proj.get_base_layer().weight = orig_weights_out
                    base_layer.out_proj.merge(adapter_names=[active_adapter])
                else:
                    weight_merged = base_layer.in_proj_weight.data.detach() + self.get_delta_weight(active_adapter)
                    del base_layer.in_proj_weight
                    base_layer.in_proj_weight = weight_merged
                    weight_merged = base_layer.out_proj.weight.data.detach() + base_layer.out_proj.get_delta_weight(active_adapter)
                    del base_layer.out_proj.get_base_layer().weight
                    base_layer.out_proj.get_base_layer().weight = weight_merged
                    base_layer.out_proj.merge(adapter_names=[active_adapter])
                self.merged_adapters.append(active_adapter)

    def unmerge(self) ->None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                old_weight = base_layer.in_proj_weight.data - self.get_delta_weight(active_adapter)
                del base_layer.in_proj_weight
                base_layer.register_parameter('in_proj_weight', nn.Parameter(old_weight, requires_grad=False))
                old_weight = base_layer.out_proj.base_layer.weight.data - base_layer.out_proj.get_delta_weight(active_adapter)
                del base_layer.out_proj.base_layer.weight
                base_layer.out_proj.base_layer.register_parameter('weight', nn.Parameter(old_weight, requires_grad=False))
        self.get_base_layer().out_proj.unmerge()

    def unload_and_optionally_merge_module(self, merge: 'bool', safe_merge: 'bool', adapter_names: 'Optional[list[str]]') ->nn.MultiheadAttention:
        """
        Merging and unloading of the MultiheadAttention module

        This requires an extra step for MultiheadAttention, which is why there is this special method instead of
        relying on the normal merge_and_unload code path.
        """
        if merge:
            self.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        base_layer = self.get_base_layer()
        weight = base_layer.in_proj_weight
        del base_layer.in_proj_weight
        base_layer.register_parameter('in_proj_weight', nn.Parameter(weight.data, requires_grad=weight.requires_grad))
        out_proj_layer = base_layer.out_proj.get_base_layer()
        weight = out_proj_layer.weight
        del out_proj_layer.weight
        out_proj_layer.register_parameter('weight', nn.Parameter(weight.data, requires_grad=weight.requires_grad))
        base_layer.out_proj = out_proj_layer
        return base_layer

    def get_delta_weight(self, adapter) ->torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype
        cast_to_fp32 = device.type == 'cpu' and dtype == torch.float16
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        output_tensor = weight_B @ weight_A * self.scaling[adapter]
        if cast_to_fp32:
            output_tensor = output_tensor
            self.lora_A[adapter].weight.data = weight_A
            self.lora_B[adapter].weight.data = weight_B
        return output_tensor

    def _check_forward_args(self, x, *args, **kwargs):
        if 'adapter_names' in kwargs:
            raise TypeError(f'lora.{self.__class__.__name__} does not support mixed adapter batches.')
        super()._check_forward_args(x, *args, **kwargs)

    def forward(self, x: 'torch.Tensor', *args: Any, **kwargs: Any) ->torch.Tensor:
        previous_dtype = x.dtype
        self._check_forward_args(x, *args, **kwargs)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            out_proj = self.get_base_layer().out_proj
            if out_proj.active_adapters != self.active_adapters:
                cls_name = self.get_base_layer().__class__.__name__
                raise ValueError(f"The out_proj layer of {cls_name} has merged layers but {cls_name} itself doesn't; please ensure that either both or none have merged layers")
            active_adapters = [a for a in self.active_adapters if a in self.lora_A]
            try:
                self.merge(adapter_names=active_adapters)
                result = self.base_layer(x, *args, **kwargs)
            finally:
                self.unmerge()
        result = result[0], result[1] if result[1] is not None else result[1]
        return result

    @skip_init_on_device
    def _restore_weights(self):
        base_layer = self.get_base_layer()
        weight = base_layer.in_proj_weight
        del base_layer.in_proj_weight
        base_layer.register_parameter('in_proj_weight', nn.Parameter(weight.data, requires_grad=weight.requires_grad))
        base_layer = base_layer.out_proj.get_base_layer()
        weight = base_layer.weight
        del base_layer.weight
        base_layer.register_parameter('weight', nn.Parameter(weight.data, requires_grad=weight.requires_grad))

    def state_dict(self, *args, **kwargs):
        self._restore_weights()
        return super().state_dict(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        self._restore_weights()
        return super().named_modules(*args, **kwargs)

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.' + rep


def dispatch_default(target: 'torch.nn.Module', adapter_name: 'str', lora_config: 'LoraConfig', **kwargs) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop('fan_in_fan_out', None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Conv1d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv1d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.MultiheadAttention):
        kwargs.update(lora_config.loftq_config)
        new_module = MultiheadAttention(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting fan_in_fan_out to False.')
            kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True.')
            kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)
    return new_module


@lru_cache
def is_eetq_available():
    return importlib.util.find_spec('eetq') is not None


def dispatch_eetq(target: 'torch.nn.Module', adapter_name: 'str', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_eetq_available() and isinstance(target_base_layer, EetqLinear):
        new_module = EetqLoraLinear(target, adapter_name, **kwargs)
        target.weight = target_base_layer.weight
        if hasattr(target, 'bias'):
            target.bias = target_base_layer.bias
    return new_module


class QuantLinear(torch.nn.Module, LoraLayer):

    def __init__(self, base_layer, adapter_name: 'str', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, init_lora_weights: 'bool'=True, use_rslora: 'bool'=False, use_dora: 'bool'=False, lora_bias: 'bool'=False, **kwargs):
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA yet, please set it to False')
        self.quant_linear_module = base_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora, lora_bias=lora_bias)

    def forward(self, x: 'torch.Tensor'):
        result = self.quant_linear_module(x)
        if self.disable_adapters:
            return result
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x
            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output
            output = output * scaling
            result += output
        return result

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.' + rep


@lru_cache
def is_optimum_available() ->bool:
    return importlib.util.find_spec('optimum') is not None


@lru_cache
def is_gptqmodel_available():
    if importlib.util.find_spec('gptqmodel') is not None:
        GPTQMODEL_MINIMUM_VERSION = packaging.version.parse('1.7.0')
        OPTIMUM_MINIMUM_VERSION = packaging.version.parse('1.23.99')
        version_gptqmodel = packaging.version.parse(importlib_metadata.version('gptqmodel'))
        if GPTQMODEL_MINIMUM_VERSION <= version_gptqmodel:
            if is_optimum_available():
                version_optimum = packaging.version.parse(importlib_metadata.version('optimum'))
                if OPTIMUM_MINIMUM_VERSION <= version_optimum:
                    return True
                else:
                    raise ImportError(f'gptqmodel requires optimum version {OPTIMUM_MINIMUM_VERSION} or higher. Found version {version_optimum}, but only versions above {OPTIMUM_MINIMUM_VERSION} are supported')
            else:
                raise ImportError(f'gptqmodel requires optimum version {OPTIMUM_MINIMUM_VERSION} or higher to be installed.')
        else:
            raise ImportError(f'Found an incompatible version of gptqmodel. Found version {version_gptqmodel}, but only versions above {GPTQMODEL_MINIMUM_VERSION} are supported')


def get_gptqmodel_quant_linear(gptq_quantization_config, device_map=None):
    """
    Get the right GPTQQuantLinear class based on the quantization config file
    """
    if gptq_quantization_config is None:
        return None
    if not is_gptqmodel_available():
        return None
    desc_act = gptq_quantization_config.desc_act
    group_size = gptq_quantization_config.group_size
    bits = gptq_quantization_config.bits
    checkpoint_format = gptq_quantization_config.checkpoint_format if hasattr(gptq_quantization_config, 'checkpoint_format') else 'gptq'
    sym = gptq_quantization_config.sym
    meta = gptq_quantization_config.meta if hasattr(gptq_quantization_config, 'meta') else None
    QuantLinear = hf_select_quant_linear(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, device_map=device_map, checkpoint_format=checkpoint_format, meta=meta, backend='auto_trainable')
    return QuantLinear


def dispatch_gptq(target: 'torch.nn.Module', adapter_name: 'str', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    cfg = kwargs.get('gptq_quantization_config', None)
    if is_gptqmodel_available():
        device_map = kwargs.get('device_map', None)
        quant_linear = get_gptqmodel_quant_linear(cfg, device_map=device_map)
    else:
        quant_linear = get_auto_gptq_quant_linear(cfg)
    if quant_linear is not None and isinstance(target_base_layer, quant_linear):
        new_module = QuantLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.qweight
    return new_module


@lru_cache
def is_hqq_available():
    return importlib.util.find_spec('hqq') is not None


def dispatch_hqq(target: 'torch.nn.Module', adapter_name: 'str', **kwargs):
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_hqq_available() and isinstance(target_base_layer, HQQLinear):
        new_module = HqqLoraLinear(target_base_layer, adapter_name, **kwargs)
    return new_module


class LoraParallelLinear(nn.Module, LoraLayer):
    """
    When the target layer parallel_linear is RowParallelLinear, in order to keep the input and output shapes
    consistent, we need to split the lora matrix A into rows, and the lora_B at this time should be a complete linear
    layer; In the same way, when the target layer is ColumnParallelLinear, we perform column segmentation on lora_B,
    while lora_A is still a complete linear layer.
    """

    def __init__(self, base_layer, adapter_name: 'str', backend, r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, fan_in_fan_out: 'bool'=False, is_target_conv_1d_layer: 'bool'=False, init_lora_weights: 'Union[bool, str]'=True, use_rslora: 'bool'=False, use_dora: 'bool'=False, lora_bias: 'bool'=False, **kwargs):
        if lora_bias:
            raise ValueError(f'{self.__class__.__name__} does not support lora_bias yet, set it to False')
        super().__init__()
        LoraLayer.__init__(self, base_layer=base_layer, **kwargs)
        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA yet, please set it to False')
        self.backend = backend
        self.is_parallel_a = isinstance(base_layer, backend.RowParallelLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        megatron_config = kwargs['megatron_config']
        parallel_linear_kwargs = {'megatron_config': megatron_config}
        init_method = init.xavier_normal_
        if hasattr(megatron_config, 'init_method'):
            init_method = megatron_config.init_method
        input_is_parallel = True
        gather_output = False
        if isinstance(base_layer, self.backend.RowParallelLinear):
            input_is_parallel = base_layer.input_is_parallel
        else:
            gather_output = base_layer.gather_output
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora, init_method=init_method, input_is_parallel=input_is_parallel, gather_output=gather_output, **parallel_linear_kwargs)
        if is_target_conv_1d_layer:
            raise ValueError(f'{self.__class__.__name__} does not support target_conv_1d_layer yet, please set it to False')
        self.is_target_conv_1d_layer = False

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora=False, init_method=init.xavier_normal_, input_is_parallel=True, gather_output=False, **parallel_linear_kwargs):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer
        megatron_config = parallel_linear_kwargs['megatron_config']
        megatron_config.params_dtype = torch.float32
        if self.is_parallel_a:
            lora_a = self.backend.RowParallelLinear(input_size=self.in_features, output_size=r, bias=False, input_is_parallel=input_is_parallel, skip_bias_add=True, init_method=init_method, config=megatron_config)
            lora_b = nn.Linear(in_features=r, out_features=self.out_features, bias=False, dtype=torch.float32)
        else:
            lora_a = nn.Linear(in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32)
            lora_b = self.backend.ColumnParallelLinear(input_size=r, output_size=self.out_features, bias=False, gather_output=gather_output, init_method=init_method, config=megatron_config)
        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith('pissa'):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith('corda'):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == 'olora':
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == 'loftq':
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False
        self.set_adapter(self.active_adapters)

    def forward(self, x: 'torch.Tensor', *args: Any, **kwargs: Any):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop('adapter_names', None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result, bias = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise ValueError(f'{self.__class__.__name__} does not support mixed_batch_forward yet.')
        elif self.merged:
            result, bias = self.base_layer(x, *args, **kwargs)
        else:
            result, bias = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x
                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, torch.nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None
                    result = result + self.lora_magnitude_vector[active_adapter](x, lora_A=lora_A, lora_B=lora_B, scaling=scaling, base_layer=self.get_base_layer(), base_result=base_result)
            result = result
        return result, bias

    def merge(self, safe_merge: 'bool'=False, adapter_names: 'Optional[list[str]]'=None) ->None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        weight_norm = self.lora_magnitude_vector[active_adapter].get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1).detach()
                        self._cache_store(f'{active_adapter}-weight_norm', weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        weight_norm = self.lora_magnitude_vector[active_adapter].get_weight_norm(base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1).detach()
                        self._cache_store(f'{active_adapter}-weight_norm', weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight
                self.merged_adapters.append(active_adapter)

    def unmerge(self) ->None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f'{active_adapter}-weight_norm')
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) ->torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype
        cast_to_fp32 = device.type == 'cpu' and (dtype == torch.float16 or dtype == torch.bfloat16)
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        if cast_to_fp32:
            output_tensor = output_tensor
            self.lora_A[adapter].weight.data = weight_A
            self.lora_B[adapter].weight.data = weight_B
        return output_tensor

    def __repr__(self) ->str:
        rep = super().__repr__()
        return 'lora.' + rep


def dispatch_megatron(target: 'torch.nn.Module', adapter_name: 'str', lora_config, **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if lora_config.megatron_config:
        megatron_core = importlib.import_module(lora_config.megatron_core)
    else:
        megatron_core = None
    if megatron_core and isinstance(target_base_layer, (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear)):
        megatron_kwargs = kwargs.copy()
        megatron_config = lora_config.megatron_config
        if isinstance(megatron_config, dict):
            transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
            megatron_config = transformer_config_class(**lora_config.megatron_config)
        megatron_kwargs['megatron_config'] = megatron_config
        if megatron_kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to True but the target module is `ColumnParallelLinear` or `RowParallelLinear`. Setting fan_in_fan_out to False.')
            megatron_kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = False
        new_module = LoraParallelLinear(base_layer=target, adapter_name=adapter_name, backend=megatron_core.tensor_parallel, **megatron_kwargs)
    return new_module


class TorchaoLoraLinear(Linear):
    """LoRA layer implementation for Linear layers using torchao data"""

    def __init__(self, *args, get_apply_tensor_subclass, **kwargs):
        if kwargs.get('lora_bias', False):
            raise ValueError(f'{self.__class__.__name__} does not support lora_bias yet, set it to False')
        super().__init__(*args, **kwargs)
        self.get_apply_tensor_subclass = get_apply_tensor_subclass
        self._check_dtype_supported()

    def _check_dtype_supported(self):
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        if hasattr(weight, 'tensor_impl') and weight.tensor_impl.data.dtype != torch.int8 or hasattr(weight, 'layout_tensor') and weight.layout_tensor.data.dtype != torch.int8:
            raise ValueError(f'{type(self).__name__} only supports int8 weights for now.')

    def merge(self, safe_merge: 'bool'=False, adapter_names: 'Optional[list[str]]'=None) ->None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        self._check_dtype_supported()
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        for active_adapter in adapter_names:
            try:
                weight = weight.dequantize()
            except NotImplementedError as exc:
                msg = f'Weights of type {type(weight).__name__} do not support dequantization (yet), which is needed to support merging.'
                raise NotImplementedError(msg) from exc
            if safe_merge and not torch.isfinite(weight).all():
                raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
            weight += self.get_delta_weight(active_adapter)
            del base_layer.weight
            base_layer.weight = weight
            quantize_(base_layer, self.get_apply_tensor_subclass())
            del weight
            self.merged_adapters.append(active_adapter)

    def unmerge(self) ->None:
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_A.keys():
                continue
            base_layer = self.get_base_layer()
            weight = base_layer.weight
            try:
                weight = weight.dequantize()
            except NotImplementedError as exc:
                msg = f'Weights of type {type(weight).__name__} do not support dequantization (yet), which is needed to support unmerging.'
                raise NotImplementedError(msg) from exc
            weight -= self.get_delta_weight(active_adapter)
            del base_layer.weight
            base_layer.weight = weight
            quantize_(base_layer, self.get_apply_tensor_subclass())
            del weight

    def __repr__(self) ->str:
        rep = super().__repr__()
        return rep.replace('lora.Linear', f'lora.{self.__class__.__name__}')


@lru_cache
def is_torchao_available():
    if importlib.util.find_spec('torchao') is None:
        return False
    TORCHAO_MINIMUM_VERSION = packaging.version.parse('0.4.0')
    try:
        torchao_version = packaging.version.parse(importlib_metadata.version('torchao'))
    except importlib_metadata.PackageNotFoundError:
        return False
    if torchao_version < TORCHAO_MINIMUM_VERSION:
        raise ImportError(f'Found an incompatible version of torchao. Found version {torchao_version}, but only versions above {TORCHAO_MINIMUM_VERSION} are supported')
    return True


def dispatch_torchao(target: 'torch.nn.Module', adapter_name: 'str', lora_config: 'LoraConfig', **kwargs: Any) ->Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if not hasattr(target_base_layer, 'weight'):
        return new_module
    if not is_torchao_available():
        return new_module
    if isinstance(target_base_layer.weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor)):
        new_module = TorchaoLoraLinear(target, adapter_name, **kwargs)
    return new_module


def get_pattern_key(pattern_keys, key_to_match):
    """Match a substring of key_to_match in pattern keys"""
    return next(filter(lambda key: re.match(f'.*\\.{key}$', key_to_match), pattern_keys), key_to_match)


def str_to_bool(value: 'str') ->int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif value in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError(f'invalid truth value {value}')


def check_file_exists_on_hf_hub(repo_id: 'str', filename: 'str', **kwargs) ->Optional[bool]:
    """Check if a file exists on HF Hub, if check was not successful returns None instead of erroring.

    Respect offline mode if set.

    """
    exists: 'Optional[bool]' = None
    if str_to_bool(os.environ.get('HF_HUB_OFFLINE', '0')):
        return exists
    try:
        exists = file_exists(repo_id, filename, **kwargs)
    except (HFValidationError, EntryNotFoundError):
        pass
    except Exception as e:
        warnings.warn(f'Unable to fetch remote file due to the following error {e} - silently ignoring the lookup for the file {filename} in {repo_id}.')
    return exists


def get_embedding_layer_name(model, layer, is_embedding_in_target_modules):
    """Get the name of the embedding module for a given layer."""
    for name, module in model.named_modules():
        if not is_embedding_in_target_modules and module == layer or module == getattr(layer, 'base_layer', None):
            return name
    return None


def has_valid_embedding_base_layer(layer):
    """Check if the layer has an embedding base layer"""
    return hasattr(layer, 'base_layer') and isinstance(layer.base_layer, (torch.nn.Linear, torch.nn.Embedding))


def get_peft_model_state_dict(model, state_dict=None, adapter_name='default', unwrap_compiled=False, save_embedding_layers='auto'):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be returned.
        unwrap_compiled (`bool`, *optional*, defaults to `False`):
            Whether to unwrap the model if torch.compile was used.
        save_embedding_layers (`Union[bool, str]`, , *optional*, defaults to `auto`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common embedding
            layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available. Based on it
            sets the boolean flag. This only works for 🤗 transformers models.
    """
    if unwrap_compiled:
        model = getattr(model, '_orig_mod', model)
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
        elif bias == 'all':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k or 'bias' in k}
        elif bias == 'lora_only':
            to_return = {}
            for k in state_dict:
                if 'lora_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('lora_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if 'lora_' in k and adapter_name in k or 'bias' in k}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f'.{adapter_name}', ''): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)
        if config.use_dora:
            new_dora_suffix = f'lora_magnitude_vector.{adapter_name}.weight'

            def renamed_dora_weights(k):
                if k.endswith(new_dora_suffix):
                    k = k[:-7]
                return k
            to_return = {renamed_dora_weights(k): v for k, v in to_return.items()}
    elif config.peft_type == PeftType.BOFT:
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'boft_' in k}
        elif bias == 'all':
            to_return = {k: state_dict[k] for k in state_dict if 'boft_' in k or 'bias' in k}
        elif bias == 'boft_only':
            to_return = {}
            for k in state_dict:
                if 'boft_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('boft_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split('.')[-1].startswith('adaption_')}
    elif config.is_prompt_learning:
        to_return = {}
        if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            to_return['prefix_task_cols'] = model.prompt_encoder[adapter_name].prefix_task_cols
            to_return['prefix_task_rows'] = model.prompt_encoder[adapter_name].prefix_task_rows
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        elif config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return['prompt_embeddings'] = prompt_embeddings
    elif config.peft_type == PeftType.VERA:
        vera_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        to_return = {k: state_dict[k] for k in state_dict if vera_prefix in k}
        if config.save_projection:
            if f'base_model.vera_A.{adapter_name}' not in state_dict:
                raise ValueError('Model was initialised to not save vera_A and vera_B but config now specifies to save projection! Set `config.save_projection` to `False`.')
            to_return['base_model.vera_A.' + adapter_name] = state_dict['base_model.vera_A.' + adapter_name]
            to_return['base_model.vera_B.' + adapter_name] = state_dict['base_model.vera_B.' + adapter_name]
    elif config.peft_type == PeftType.XLORA:
        to_return = {k: state_dict[k] for k in state_dict if 'internal_xlora_classifier' in k}
    elif config.peft_type == PeftType.VBLORA:
        to_return = {}
        if config.num_vectors < 2 ** 8:
            indices_dtype = torch.uint8
        elif config.num_vectors < 2 ** 15:
            indices_dtype = torch.int16
        elif config.num_vectors < 2 ** 31:
            indices_dtype = torch.int32
        else:
            indices_dtype = torch.int64
        if config.save_only_topk_weights:
            for k in state_dict:
                if 'vblora_logits' in k:
                    logits, indices = state_dict[k].topk(config.topk)
                    to_return.update({(k + '_topk_indices'): indices})
                    to_return.update({(k + '_topk_weights'): torch.softmax(logits, dim=-1)[:, :, :-1].contiguous()})
        else:
            to_return = {k: state_dict[k] for k in state_dict if 'vblora_logits' in k}
        to_return['base_model.vblora_vector_bank.' + adapter_name] = state_dict['base_model.vblora_vector_bank.' + adapter_name]
    elif config.peft_type in list(PeftType):
        prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        to_return = {k: state_dict[k] for k in state_dict if prefix in k}
    else:
        raise ValueError(f'Unknown PEFT type passed: {config.peft_type}')
    if getattr(model, 'modules_to_save', None) is not None:
        for key, value in state_dict.items():
            if any(f'{module_name}.modules_to_save.{adapter_name}' in key for module_name in model.modules_to_save):
                to_return[key.replace('modules_to_save.', '')] = value
    is_embedding_in_target_modules = False
    if save_embedding_layers == 'auto' and hasattr(config, 'target_modules') and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES):
        warnings.warn('Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.')
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == 'auto':
        vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', None)
        model_id = getattr(config, 'base_model_name_or_path', None)
        has_base_config = False
        if model_id is not None:
            local_config_exists = os.path.exists(os.path.join(model_id, 'config.json'))
            exists = local_config_exists or check_file_exists_on_hf_hub(model_id, 'config.json')
            if exists is None:
                warnings.warn(f'Could not find a config file in {model_id} - will assume that the vocabulary was not modified.')
                has_base_config = False
            else:
                has_base_config = exists
        if vocab_size and model_id and has_base_config and vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size:
            warnings.warn('Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.')
            save_embedding_layers = True
        else:
            save_embedding_layers = False
    if save_embedding_layers and hasattr(model, 'get_input_embeddings'):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                embedding_module_name = get_embedding_layer_name(model, layer, is_embedding_in_target_modules)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn('Could not identify embedding layer(s) because the model is not a 🤗 transformers model.')
    to_return = {k.replace(f'.{adapter_name}', ''): v for k, v in to_return.items()}
    return to_return


def get_quantization_config(model: 'torch.nn.Module', method: 'str'):
    """
    Get the quantization config of the related quantization method
    """
    if hasattr(model, 'config') and hasattr(model.config, 'quantization_config') and getattr(model, 'quantization_method', None) == method:
        return model.config.quantization_config
    return None


@lru_cache
def is_bnb_available() ->bool:
    return importlib.util.find_spec('bitsandbytes') is not None


@lru_cache
def is_bnb_4bit_available() ->bool:
    if not is_bnb_available():
        return False
    return hasattr(bnb.nn, 'Linear4bit')


def magnitude_prune(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor', density: 'float') ->torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`): The fraction of values to preserve. Should be in [0,1].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='magnitude') for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def clone_module(module: 'nn.Module', share_weights=False):
    """Clone a module in a pytorch model.

    Clones a module of a model, optionally sharing all the parameters between the original and the clone. Simplifies
    reusing a module when manipulating the architecture of a model.
    """
    clone = copy.deepcopy(module)

    def _share_weights(src: 'nn.Module', dst: 'nn.Module'):
        for name, param in src.named_parameters(recurse=False):
            dst.register_parameter(name, param)
    if share_weights:
        for name, submodule in module.named_modules():
            _share_weights(submodule, clone.get_submodule(name))
    return clone


def replicate_layers(model: 'nn.Module', layer_map: 'list[tuple[int, int]]'):
    """Replicate layers in a transfomer model with weight sharing.

    This function looks for a module list attribute at model[(.model)*].layers and replicates the layers in the module
    list according to the layer map. For example the map `[[0, 4], [2, 5]]` will take the set of layers `[0, 1, 2, 3,
    4]` and replace them with a module list containing `[0, 1, 2, 3, 2, 3, 4]`.
    """
    while hasattr(model, 'model'):
        model = model.model
    if hasattr(model, 'bert'):
        model = model.bert
    model_type = None
    layers: 'nn.ModuleList' = None
    if hasattr(model, 'layers'):
        model_type = 'llama'
        layers = model.layers
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        model_type = 'bert'
        layers = model.encoder.layer
    elif hasattr(model, 'h'):
        model_type = 'falcon'
        layers = model.h
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError('Could not locate the layers attribute in the model. Expected Llama, Bert or Falcon compatible architectures.')
    new_layers = []
    for start, end in layer_map:
        for i in range(start, end):
            current_idx = len(new_layers)
            new_layers.append(clone_module(layers[i], share_weights=True))
            for submodule in new_layers[-1].modules():
                if hasattr(submodule, 'layer_idx'):
                    submodule.layer_idx = current_idx
    layers = nn.ModuleList(new_layers)
    if model_type == 'llama':
        model.layers = layers
    elif model_type == 'bert':
        model.encoder.layer = layers
    elif model_type == 'falcon':
        model.h = layers
    else:
        raise ValueError('Unexpected model type, need to handle post-processing of layers.')
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(new_layers)


def task_arithmetic(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor') ->torch.Tensor:
    """
    Merge the task tensors using `task arithmetic`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = torch.stack(task_tensors, dim=0)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def ties(task_tensors: 'List[torch.Tensor]', weights: 'torch.Tensor', density: 'float', majority_sign_method: "Literal['total', 'frequency']"='total') ->torch.Tensor:
    """
    Merge the task tensors using `ties`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.
        density (`float`):The fraction of values to preserve. Should be in [0,1].
        majority_sign_method (`str`):
            The method to use to get the majority sign mask. Should be one of ["total", "frequency"].

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = [prune(tensor, density, method='magnitude') for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


class TemperatureScaledSoftmax(nn.Module):

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return self.softmax(scaled_logits)


class XLoraClassifier(nn.Module):
    """
    A classifier to select LoRA layers for XLora.
    """

    def __init__(self, model: 'nn.Module', config: 'XLoraConfig', n_classes: 'int', n_layers: 'int', device: 'torch.device'):
        """
        Construct an X-LoRA classifier from a model, config and some metadata. Note that n_layers is the number of LoRA
        adapter layers, not the number of model layers.
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.config = config
        self.log_scalings = []
        self.softmax = TemperatureScaledSoftmax(temperature=self.config.softmax_temperature)
        self.override_scaling_pass_value: 'Number' = config.scaling_pass_value
        self.scalings_logging = False
        self.dtype = next(model.parameters()).dtype
        add_dropout = config.xlora_dropout_p > 0.0
        layers = []
        if self.config.xlora_depth == 1:
            if config.layerwise_scalings:
                last = nn.Linear(config.hidden_size, n_classes * n_layers, bias=True).to(device)
            else:
                last = nn.Linear(config.hidden_size, n_classes, bias=True).to(device)
        else:
            if self.config.xlora_depth <= 0:
                raise ValueError('X-LoRA depth must be strictly positive.')
            layers.append(nn.Linear(config.hidden_size, config.xlora_size, bias=True).to(device))
            layers.append(nn.ReLU())
            if add_dropout:
                layers.append(nn.Dropout(p=config.xlora_dropout_p))
            for _ in range(config.xlora_depth - 2):
                layers.append(nn.Linear(config.xlora_size, config.xlora_size, bias=True).to(device))
                layers.append(nn.ReLU())
                if add_dropout:
                    layers.append(nn.Dropout(p=config.xlora_dropout_p))
            if config.layerwise_scalings:
                last = nn.Linear(config.xlora_size, n_classes * n_layers, bias=True).to(device)
            else:
                last = nn.Linear(config.xlora_size, n_classes, bias=True).to(device)
        self.layers = nn.Sequential(*layers, last)

    def make_dummy_scalings(self, input_ids: 'Optional[torch.LongTensor]'=None, inputs_embeds: 'Optional[torch.FloatTensor]'=None, *args, **kwargs) ->torch.Tensor:
        """
        Make some dummy scalings for the scalings pass (the one to get the logits for the X-LoRA classifier). These are
        of shape (batch_size, seq_len, n_layers, n_classes) and filled with the override scalings pass value. Note that
        n_layers is the number of LoRA adapter layers, not the number of model layers.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            seq_len = input_ids.shape[1]
        else:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            seq_len = inputs_embeds.shape[1]
        return torch.full((batch_size, seq_len, self.n_layers, self.n_classes), self.override_scaling_pass_value)

    def forward(self, result, input_ids: 'Optional[torch.LongTensor]'=None, inputs_embeds: 'Optional[torch.FloatTensor]'=None, *args, **kwargs) ->torch.Tensor:
        """
        Using the hidden states of the model, predict `n_classes` LoRA alpha values. Returns the scalings.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
        else:
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
        hidden_states = result.hidden_states
        hidden_state = hidden_states[-1]
        logits = self.layers.forward(hidden_state)
        if not self.config.layerwise_scalings:
            logits = logits.unsqueeze(2)
            logits = logits.expand(-1, -1, self.n_layers, -1)
        scalings = logits.reshape(batch_size, seq_len, self.n_layers, self.n_classes)
        if self.config.enable_softmax:
            scalings = self.softmax(scalings)
        if self.scalings_logging:
            self.log_scalings.append(scalings)
        return scalings

    def _get_bucketed_scalings(self) ->dict[int, tuple[list[int], list[torch.Tensor]]]:
        """
        Returns bucketed scalings, bucketed by seq_len. Each value consists of the positions (the first) and the
        associated tensors. The positions are paired with the associated tensors and give the position in the scaling
        log. Each scaling is a tensor of shape (batch_size, seq_len, n_layers, n_classes)).
        """
        seqlens_map: 'dict[int, tuple[list[int], list[torch.Tensor]]]' = {}
        for i, scaling in enumerate(self.log_scalings):
            seq_len = scaling.shape[1]
            if seq_len not in seqlens_map:
                seqlens_map[seq_len] = [i], [scaling]
            else:
                seqlens_map[seq_len][0].append(i)
                seqlens_map[seq_len][1].append(scaling)
        return seqlens_map

    def _set_override_scaling_pass_value(self, value: 'Union[Number, None]'):
        if value is None:
            self.override_scaling_pass_value = 1 / self.n_classes
        else:
            self.override_scaling_pass_value = value
        self.config.scaling_pass_value = self.override_scaling_pass_value

