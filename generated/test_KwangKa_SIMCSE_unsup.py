import sys
_module = sys.modules[__name__]
del sys
CSECollator = _module
SimCSE = _module
SimCSERetrieval = _module
eval_unsup = _module
test_unsup = _module
train_unsup = _module

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


import torch.nn as nn


import numpy as np


import torch


import logging


import torch.nn.functional as F


import scipy.stats


from torch.utils.data import DataLoader


class SimCSE(nn.Module):

    def __init__(self, pretrained='hfl/chinese-bert-wwm-ext', pool_type='cls', dropout_prob=0.3):
        super().__init__()
        conf = BertConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        assert pool_type in ['cls', 'pooler'], 'invalid pool_type: %s' % pool_type
        self.pool_type = pool_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.pool_type == 'cls':
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == 'pooler':
            output = output.pooler_output
        return output

