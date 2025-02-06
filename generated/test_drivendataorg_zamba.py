import sys
_module = sys.modules[__name__]
del sys
generate_api_reference = _module
conftest = _module
test_cli = _module
test_config = _module
test_datamodule = _module
test_densepose = _module
test_depth = _module
test_filter_frames = _module
test_instantiate_model = _module
test_load_video_frames = _module
test_megadetector_lite_yolox = _module
test_metadata = _module
test_metrics = _module
test_model_manager = _module
test_npy_cache = _module
test_transforms = _module
test_zamba_video_classification_lightning_module = _module
zamba = _module
cli = _module
data = _module
metadata = _module
video = _module
exceptions = _module
metrics = _module
models = _module
config = _module
densepose = _module
densepose_manager = _module
depth_estimation = _module
depth_manager = _module
efficientnet_models = _module
model_manager = _module
publish_models = _module
registry = _module
slowfast_models = _module
utils = _module
object_detection = _module
yolox = _module
megadetector_lite_yolox = _module
yolox_model = _module
pytorch = _module
dataloaders = _module
finetuning = _module
layers = _module
transforms = _module
utils = _module
pytorch_lightning = _module
utils = _module
settings = _module
version = _module

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


import random


import string


from typing import Optional


from typing import Union


import pandas as pd


import torch


import numpy as np


from enum import Enum


from typing import Dict


import torch.utils


import torch.utils.data


from torchvision import transforms


from torchvision.transforms import Resize


from torch import nn


from typing import Tuple


import copy


from functools import lru_cache


from typing import List


import warnings


import torchvision.datasets.video_utils


from torchvision.datasets.vision import VisionDataset


import torchvision.transforms.transforms


import itertools


from torchvision.transforms import Normalize


from sklearn.metrics import f1_score


from sklearn.metrics import top_k_accuracy_score


from sklearn.metrics import accuracy_score


import torch.nn.functional as F


from torchvision.transforms import transforms


def _stack_tups(tuples, stack_dim=1):
    """Stack tuple of tensors along `stack_dim`

    NOTE: vendored (with minor adaptations) from fastai:
    https://github.com/fastai/fastai/blob/4b0785254fdece1a44859956b6e54eedb167a97e/fastai/layers.py#L505-L507

    Updates:
        -  use `range` rather than fastai `range_of`
    """
    return tuple(torch.stack([t[i] for t in tuples], dim=stack_dim) for i in range(len(tuples[0])))


class TimeDistributed(torch.nn.Module):
    """Applies `module` over `tdim` identically for each step, use `low_mem` to compute one at a time.

    NOTE: vendored (with minor adaptations) from fastai:
    https://github.com/fastai/fastai/blob/4b0785254fdece1a44859956b6e54eedb167a97e/fastai/layers.py#L510-L544

    Updates:
     - super.__init__() in init
     - assign attributes in init
     - inherit from torch.nn.Module rather than fastai.Module
    """

    def __init__(self, module, low_mem=False, tdim=1):
        super().__init__()
        self.low_mem = low_mem
        self.tdim = tdim
        self.module = module

    def forward(self, *tensors, **kwargs):
        """input x with shape:(bs,seq_len,channels,width,height)"""
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*tensors, **kwargs)
        else:
            inp_shape = tensors[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(*[x.view(bs * seq_len, *x.shape[2:]) for x in tensors], **kwargs)
        return self.format_output(out, bs, seq_len)

    def low_mem_forward(self, *tensors, **kwargs):
        """input x with shape:(bs,seq_len,channels,width,height)"""
        seq_len = tensors[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in tensors]
        out = []
        for i in range(seq_len):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        if isinstance(out[0], tuple):
            return _stack_tups(out, stack_dim=self.tdim)
        return torch.stack(out, dim=self.tdim)

    def format_output(self, out, bs, seq_len):
        """unstack from batchsize outputs"""
        if isinstance(out, tuple):
            return tuple(out_i.view(bs, seq_len, *out_i.shape[1:]) for out_i in out)
        return out.view(bs, seq_len, *out.shape[1:])

    def __repr__(self):
        return f'TimeDistributed({self.module})'


class ConvertTHWCtoCTHW(torch.nn.Module):
    """Convert tensor from (0:T, 1:H, 2:W, 3:C) to (3:C, 0:T, 1:H, 2:W)"""

    def forward(self, vid: 'torch.Tensor') ->torch.Tensor:
        return vid.permute(3, 0, 1, 2)


class ConvertTHWCtoTCHW(torch.nn.Module):
    """Convert tensor from (T, H, W, C) to (T, C, H, W)"""

    def forward(self, vid: 'torch.Tensor') ->torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertTCHWtoCTHW(torch.nn.Module):
    """Convert tensor from (T, C, H, W) to (C, T, H, W)"""

    def forward(self, vid: 'torch.Tensor') ->torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class ConvertHWCtoCHW(torch.nn.Module):
    """Convert tensor from (0:H, 1:W, 2:C) to (2:C, 0:H, 1:W)"""

    def forward(self, vid: 'torch.Tensor') ->torch.Tensor:
        return vid.permute(2, 0, 1)


class Uint8ToFloat(torch.nn.Module):

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        return tensor / 255.0


class VideotoImg(torch.nn.Module):

    def forward(self, vid: 'torch.Tensor') ->torch.Tensor:
        return vid.squeeze(0)


class PadDimensions(torch.nn.Module):
    """Pads a tensor to ensure a fixed output dimension for a give axis.

    Attributes:
        dimension_sizes: A tuple of int or None the same length as the number of dimensions in the
            input tensor. If int, pad that dimension to at least that size. If None, do not pad.
    """

    def __init__(self, dimension_sizes: 'Tuple[Optional[int]]'):
        super().__init__()
        self.dimension_sizes = dimension_sizes

    @staticmethod
    def compute_left_and_right_pad(original_size: 'int', padded_size: 'int') ->Tuple[int, int]:
        """Computes left and right pad size.

        Args:
            original_size (list, int): The original tensor size
            padded_size (list, int): The desired tensor size

        Returns:
           Tuple[int]: Pad size for right and left. For odd padding size, the right = left + 1
        """
        if original_size >= padded_size:
            return 0, 0
        pad = padded_size - original_size
        quotient, remainder = divmod(pad, 2)
        return quotient, quotient + remainder

    def forward(self, vid: 'torch.Tensor') ->torch.Tensor:
        padding = tuple(itertools.chain.from_iterable((0, 0) if padded_size is None else self.compute_left_and_right_pad(original_size, padded_size) for original_size, padded_size in zip(vid.shape, self.dimension_sizes)))
        return torch.nn.functional.pad(vid, padding[::-1])


class PackSlowFastPathways(torch.nn.Module):
    """Creates the slow and fast pathway inputs for the slowfast model."""

    def __init__(self, alpha: 'int'=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: 'torch.Tensor'):
        fast_pathway = frames
        slow_pathway = torch.index_select(frames, 1, torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long())
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvertHWCtoCHW,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConvertTCHWtoCTHW,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvertTHWCtoCTHW,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvertTHWCtoTCHW,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PackSlowFastPathways,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Uint8ToFloat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VideotoImg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_drivendataorg_zamba(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

