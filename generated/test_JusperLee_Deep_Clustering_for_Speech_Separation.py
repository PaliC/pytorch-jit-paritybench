import sys
_module = sys.modules[__name__]
del sys
config = _module
option = _module
create_scp = _module
AudioData = _module
data_loader = _module
dataloader = _module
logger = _module
set_logger = _module
model = _module
loss = _module
model = _module
test = _module
train = _module
trainer = _module
trainer = _module
utils = _module
stft_istft = _module
util = _module

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


from torch.nn.utils.rnn import pack_sequence


from torch.nn.utils.rnn import pad_sequence


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import numpy as np


import torch.nn as nn


from torch.nn.utils.rnn import pad_packed_sequence


from sklearn.cluster import KMeans


import logging


import time


import matplotlib.pyplot as plt


from collections import OrderedDict


class DPCL(nn.Module):
    """
        Implement of Deep Clustering
    """

    def __init__(self, num_layer=2, nfft=129, hidden_cells=600, emb_D=40, dropout=0.0, bidirectional=True, activation='Tanh'):
        super(DPCL, self).__init__()
        self.emb_D = emb_D
        self.blstm = nn.LSTM(input_size=nfft, hidden_size=hidden_cells, num_layers=num_layer, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(torch.nn, activation)()
        self.linear = nn.Linear(2 * hidden_cells if bidirectional else hidden_cells, nfft * emb_D)
        self.D = emb_D

    def forward(self, x, is_train=True):
        """
           input: 
                  for train: B x T x F
                  for test: T x F
           return: 
                  for train: B x TF x D
                  for test: TF x D
        """
        if not is_train:
            x = torch.unsqueeze(x, 0)
        x, _ = self.blstm(x)
        if is_train:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        B = x.shape[0]
        if is_train:
            x = x.view(B, -1, self.D)
        else:
            x = x.view(-1, self.D)
        return x

