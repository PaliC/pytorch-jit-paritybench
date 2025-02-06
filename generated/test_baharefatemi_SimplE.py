import sys
_module = sys.modules[__name__]
del sys
SimplE = _module
dataset = _module
main = _module
measure = _module
tester = _module
trainer = _module

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


import math


import numpy as np


import random


import torch.nn.functional as F


class SimplE(nn.Module):

    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(SimplE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.ent_h_embs = nn.Embedding(num_ent, emb_dim)
        self.ent_t_embs = nn.Embedding(num_ent, emb_dim)
        self.rel_embs = nn.Embedding(num_rel, emb_dim)
        self.rel_inv_embs = nn.Embedding(num_rel, emb_dim)
        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

    def l2_loss(self):
        return (torch.norm(self.ent_h_embs.weight, p=2) ** 2 + torch.norm(self.ent_t_embs.weight, p=2) ** 2 + torch.norm(self.rel_embs.weight, p=2) ** 2 + torch.norm(self.rel_inv_embs.weight, p=2) ** 2) / 2

    def forward(self, heads, rels, tails):
        hh_embs = self.ent_h_embs(heads)
        ht_embs = self.ent_h_embs(tails)
        th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_t_embs(tails)
        r_embs = self.rel_embs(rels)
        r_inv_embs = self.rel_inv_embs(rels)
        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

