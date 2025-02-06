import sys
_module = sys.modules[__name__]
del sys
SUR_adapter = _module
SUR_adapter_pipeline = _module
SUR_adapter_train = _module
SUR_image = _module
SUR_meta = _module
logger = _module

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


import torch.nn as nn


import inspect


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


import warnings


import logging


import math


import random


import numpy as np


import torch.nn.functional as F


import torch.utils.checkpoint


from torchvision import transforms


class Attention(nn.Module):

    def __init__(self, hidden_size=768):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = value
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        attention_weights = nn.functional.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V) + value
        result = self.output_layer(weighted_values) + weighted_values
        return result


class Adapter(nn.Module):

    def __init__(self, depth=2, adapter_weight=0.01, sd_text_size=768):
        super(Adapter, self).__init__()
        self.adapter_weight = adapter_weight
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=sd_text_size, nhead=8, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
        self.attention = Attention(hidden_size=sd_text_size)
        self.fc = nn.Linear(sd_text_size, sd_text_size)
        nn.init.zeros_(self.fc.weight)

    def forward(self, x):
        out_transformer_encoder = self.transformer_encoder(x)
        out_attention = self.attention(query=out_transformer_encoder, key=x, value=x)
        out_llm = self.fc(out_attention)
        out = self.adapter_weight * out_llm + (1 - self.adapter_weight) * x
        return out, out_transformer_encoder, out_llm


class LLMResize(nn.Module):

    def __init__(self, llm_size=6656, sd_text_size=768):
        super(LLMResize, self).__init__()
        self.fc = nn.Linear(llm_size, sd_text_size)

    def forward(self, x):
        out = self.fc(x)
        return out

