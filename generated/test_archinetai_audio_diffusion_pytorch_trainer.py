import sys
_module = sys.modules[__name__]
del sys
module_autoregressive = _module
module_autoregressive_cont = _module
module_autoregressive_gpt = _module
module_difformer = _module
module_qe = _module
module_tts = _module
module_tts_2 = _module
module_ae = _module
module_base = _module
module_diff_latent = _module
module_diff_mae = _module
module_diff_textcond = _module
module_diff_tts = _module
module_diff_tts_2 = _module
module_diff_tts_3 = _module
module_diff_tts_4 = _module
module_diff_txt_emb = _module
module_diffae = _module
module_diffqe = _module
module_qe_ar = _module
module_qe_ar2 = _module
module_qe_rq = _module
module_qe_rqtts = _module
module_upsampler = _module
utils = _module
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


from functools import reduce


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


import torch


import torchaudio


from torch import LongTensor


from torch import Tensor


from torch import nn


from torch.utils.data import DataLoader


from typing import Union


import torch.nn.functional as F


import random


import warnings


from copy import deepcopy


from math import pi


from torch import optim


from torch import einsum


import logging


class ChannelPositionalEmbedding(nn.Module):

    def __init__(self, channels: 'int', length: 'int'):
        super().__init__()
        self.length = length
        self.embedding = nn.Embedding(length, channels)

    def forward(self, x: 'Tensor') ->Tensor:
        batch_size, device = x.shape[0], x.device
        position = torch.arange(self.length, device=device)
        channels = repeat(self.embedding(position), 'n d -> b d n', b=batch_size)
        return channels

