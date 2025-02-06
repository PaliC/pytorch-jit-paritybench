import sys
_module = sys.modules[__name__]
del sys
dataset = _module
main = _module
model = _module

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


import numpy


from torch.utils import data


import torch


import torch.nn as nn


import torch.optim as optim


from sklearn.metrics import precision_score


from sklearn.metrics import f1_score


from sklearn.metrics import recall_score


from sklearn.metrics import accuracy_score


import torch.nn.functional as F


from torch import nn


class CNNModel(nn.Module):

    def __init__(self, embedding_dim, kernel_size, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self._kernel_size = kernel_size
        self._hidden_dim = hidden_dim
        self._word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._conv = nn.Conv2d(1, hidden_dim, kernel_size=(kernel_size, embedding_dim))
        self._hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sample):
        embeds = self._word_embeddings(sample)
        conv_in = embeds.view(1, 1, len(sample), -1)
        conv_out = self._conv(conv_in)
        conv_out = F.relu(conv_out)
        hidden_in = conv_out.view(self._hidden_dim, len(sample) + 1 - self._kernel_size).transpose(0, 1)
        tag_space = self._hidden2tag(hidden_in)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

