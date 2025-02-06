import sys
_module = sys.modules[__name__]
del sys
dataset = _module
irm_dataset = _module
model = _module
lstm_model = _module
train = _module
trainer = _module
base_trainer = _module
trainer = _module
util = _module
loss = _module
metrics = _module
utils = _module
visualization = _module

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


import random


import numpy as np


from torch.utils.data import Dataset


import torch


import torch.nn as nn


from torch.utils.data import DataLoader


import time


import matplotlib.pyplot as plt


from torch.nn.utils.rnn import pad_sequence


import math


from torch.utils.tensorboard import SummaryWriter


class LSTMModel(nn.Module):

    def __init__(self):
        """Construct LSTM model.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=161, hidden_size=1024, num_layers=2, batch_first=True, dropout=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=161)
        self.activation = nn.Sigmoid()

    def forward(self, ipt):
        o, h = self.lstm(ipt)
        o = self.linear(o)
        o = self.activation(o)
        return o


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LSTMModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 161])], {}),
     True),
]

class Test_haoxiangsnr_IRM_based_Speech_Enhancement_using_LSTM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

