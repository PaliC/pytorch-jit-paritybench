import sys
_module = sys.modules[__name__]
del sys
audio_data_pytorch = _module
datasets = _module
audio_web_dataset = _module
audio_web_dataset_preprocessing = _module
clotho_dataset = _module
common_voice_dataset = _module
libri_speech_dataset = _module
lj_speech_dataset = _module
meta_dataset = _module
wav_dataset = _module
youtube_dataset = _module
transforms = _module
all = _module
crop = _module
loudness = _module
mono = _module
randomcrop = _module
resample = _module
scale = _module
stereo = _module
utils = _module
setup = _module

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


from typing import Callable


from typing import List


from typing import Optional


from typing import Sequence


from typing import Union


import numpy as np


import scipy


import torch


from torch import Tensor


import re


from typing import Dict


import torchaudio


from torch import nn


from typing import Tuple


from torch.utils.data import Dataset


import math


import random


from torch.nn import functional as F


from typing import TypeVar


from torch.utils.data.dataset import Dataset


from torch.utils.data.dataset import Subset


class Crop(nn.Module):
    """Crops waveform to fixed size"""

    def __init__(self, size: 'int', start: 'int'=0) ->None:
        super().__init__()
        self.size = size
        self.start = start

    def forward(self, x: 'Tensor') ->Tensor:
        x = x[:, self.start:]
        channels, length = x.shape
        if length < self.size:
            padding_length = self.size - length
            padding = torch.zeros(channels, padding_length)
            return torch.cat([x, padding], dim=1)
        else:
            return x[:, 0:self.size]


class Loudness(nn.Module):
    """Normalizes to target loudness using BS.1770-4, requires pyloudnorm"""

    def __init__(self, sampling_rate: 'int', target: 'float'):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.target = target
        self.meter = pyln.Meter(sampling_rate)

    def forward(self, x: 'Tensor') ->Tensor:
        channels, length = x.shape
        x_numpy = x.numpy().T
        loudness = self.meter.integrated_loudness(data=x_numpy)
        if loudness == -float('inf'):
            return x
        x_normalized = pyln.normalize.loudness(data=x_numpy, input_loudness=loudness, target_loudness=self.target)
        return torch.from_numpy(x_normalized.T)


class Mono(nn.Module):
    """Overlaps all channels into one"""

    def forward(self, x: 'Tensor') ->Tensor:
        return torch.sum(x, dim=0, keepdim=True)


class RandomCrop(nn.Module):
    """Crops random chunk from the waveform"""

    def __init__(self, size: 'int') ->None:
        super().__init__()
        self.size = size

    def forward(self, x: 'Tensor') ->Tensor:
        length = x.shape[1]
        start = random.randint(0, max(length - self.size, 0))
        x = x[:, start:]
        channels, length = x.shape
        if length < self.size:
            padding_length = self.size - length
            padding = torch.zeros(channels, padding_length)
            return torch.cat([x, padding], dim=1)
        else:
            return x[:, 0:self.size]


class Resample(nn.Module):
    """Resamples frequency of waveform"""

    def __init__(self, source: 'int', target: 'int'):
        super().__init__()
        self.transform = torchaudio.transforms.Resample(orig_freq=source, new_freq=target)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.transform(x)


class Scale(nn.Module):
    """Scales waveform (change volume)"""

    def __init__(self, scale: 'float'):
        super().__init__()
        self.scale = scale

    def forward(self, x: 'Tensor') ->Tensor:
        return x * self.scale


class Stereo(nn.Module):

    def forward(self, x: 'Tensor') ->Tensor:
        shape = x.shape
        channels = shape[0]
        if len(shape) == 1:
            x = x.unsqueeze(0).repeat(2, 1)
        elif len(shape) == 2:
            if channels == 1:
                x = x.repeat(2, 1)
            elif channels > 2:
                x = x[:2, :]
        return x


T = TypeVar('T')


class AllTransform(nn.Module):

    def __init__(self, source_rate: 'Optional[int]'=None, target_rate: 'Optional[int]'=None, crop_size: 'Optional[int]'=None, random_crop_size: 'Optional[int]'=None, loudness: 'Optional[int]'=None, scale: 'Optional[float]'=None, stereo: 'bool'=False, mono: 'bool'=False):
        super().__init__()
        self.random_crop_size = random_crop_size
        message = 'Loudness requires target_rate'
        assert not exists(loudness) or exists(target_rate), message
        self.transform = nn.Sequential(Resample(source=source_rate, target=target_rate) if exists(source_rate) and exists(target_rate) and source_rate != target_rate else nn.Identity(), RandomCrop(random_crop_size) if exists(random_crop_size) else nn.Identity(), Crop(crop_size) if exists(crop_size) else nn.Identity(), Mono() if mono else nn.Identity(), Stereo() if stereo else nn.Identity(), Loudness(sampling_rate=target_rate, target=loudness) if exists(loudness) else nn.Identity(), Scale(scale) if exists(scale) else nn.Identity())

    def forward(self, x: 'Tensor') ->Tensor:
        return self.transform(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Crop,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Mono,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RandomCrop,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Resample,
     lambda: ([], {'source': 4, 'target': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Scale,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Stereo,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_archinetai_audio_data_pytorch(_paritybench_base):
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

