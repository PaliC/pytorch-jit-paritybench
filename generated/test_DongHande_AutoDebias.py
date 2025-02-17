import sys
_module = sys.modules[__name__]
del sys
arguments = _module
CausE = _module
DLA = _module
DR = _module
HeckE = _module
IPS = _module
KD_Label = _module
MF_bias = _module
MF_combine = _module
MF_uniform = _module
WMF = _module
model = _module
train_explicit = _module
train_implicit = _module
train_list = _module
utils = _module
data_loader = _module
early_stop = _module
load_dataset = _module
metrics = _module

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


import random


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import itertools


import math


import pandas as pd


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from typing import List


import copy


from enum import Enum


from enum import auto


from torch.nn import Module


import scipy.sparse as sp


from scipy.sparse import lil_matrix


class MF(nn.Module):
    """
    Base module for matrix factorization.
    """

    def __init__(self, n_user, n_item, dim=40, dropout=0, init=None):
        super().__init__()
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        if init is not None:
            self.init_embedding(init)
        else:
            self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)

    def forward(self, users, items):
        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True) + u_bias + i_bias
        return preds.squeeze(dim=-1)

    def l2_norm(self, users, items):
        users = torch.unique(users)
        items = torch.unique(items)
        l2_loss = (torch.sum(self.user_latent(users) ** 2) + torch.sum(self.item_latent(items) ** 2)) / 2
        return l2_loss


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):

    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaEmbed(MetaModule):

    def __init__(self, dim_1, dim_2):
        super().__init__()
        ignore = nn.Embedding(dim_1, dim_2)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', None)

    def forward(self):
        return self.weight

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaMF(MetaModule):
    """
    Base module for matrix factorization.
    """

    def __init__(self, n_user, n_item, dim=40, dropout=0, init=None):
        super().__init__()
        self.user_latent = MetaEmbed(n_user, dim)
        self.item_latent = MetaEmbed(n_item, dim)
        self.user_bias = MetaEmbed(n_user, 1)
        self.item_bias = MetaEmbed(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        if init is not None:
            self.init_embedding(init)
        else:
            self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)

    def forward(self, users, items):
        u_latent = self.dropout(self.user_latent.weight[users])
        i_latent = self.dropout(self.item_latent.weight[items])
        u_bias = self.user_bias.weight[users]
        i_bias = self.item_bias.weight[items]
        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True) + u_bias + i_bias
        return preds.squeeze(dim=-1)

    def l2_norm(self, users, items, unique=True):
        users = torch.unique(users)
        items = torch.unique(items)
        l2_loss = (torch.sum(self.user_latent.weight[users] ** 2) + torch.sum(self.item_latent.weight[items] ** 2)) / 2
        return l2_loss


class OneLinear(nn.Module):
    """
    linear model: r
    """

    def __init__(self, n):
        super().__init__()
        self.data_bias = nn.Embedding(n, 1)
        self.init_embedding()

    def init_embedding(self):
        self.data_bias.weight.data *= 0.001

    def forward(self, values):
        d_bias = self.data_bias(values)
        return d_bias.squeeze()


class TwoLinear(nn.Module):
    """
    linear model: u + i + r / o
    """

    def __init__(self, n_user, n_item):
        super().__init__()
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)

    def forward(self, users, items):
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        preds = u_bias + i_bias
        return preds.squeeze()


class ThreeLinear(nn.Module):
    """
    linear model: u + i + r / o
    """

    def __init__(self, n_user, n_item, n):
        super().__init__()
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.data_bias = nn.Embedding(n, 1)
        self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.data_bias.weight, mode='fan_out', a=init)
        self.data_bias.weight.data *= 0.001

    def forward(self, users, items, values):
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        d_bias = self.data_bias(values)
        preds = u_bias + i_bias + d_bias
        return preds.squeeze()


class FourLinear(nn.Module):
    """
    linear model: u + i + r + p
    """

    def __init__(self, n_user, n_item, n, n_position):
        super().__init__()
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.data_bias = nn.Embedding(n, 1)
        self.position_bias = nn.Embedding(n_position, 1)
        self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.data_bias.weight, mode='fan_out', a=init)
        self.data_bias.weight.data *= 0.001
        self.position_bias.weight.data *= 0.001

    def forward(self, users, items, values, positions):
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        d_bias = self.data_bias(values)
        p_bias = self.position_bias(positions)
        preds = u_bias + i_bias + d_bias + p_bias
        return preds.squeeze()


class Position(nn.Module):
    """
    the position parameters for DLA
    """

    def __init__(self, n_position):
        super().__init__()
        self.position_bias = nn.Embedding(n_position, 1)

    def forward(self, positions):
        return self.position_bias(positions).squeeze(dim=-1)

    def l2_norm(self, positions):
        positions = torch.unique(positions)
        return torch.sum(self.position_bias(positions) ** 2)


class MF_heckman(nn.Module):

    def __init__(self, n_user, n_item, dim=40, dropout=0, init=None):
        super().__init__()
        self.MF = MF(n_user, n_item, dim)
        self.sigma = nn.Parameter(torch.randn(1))

    def forward(self, users, items, lams):
        pred_MF = self.MF(users, items)
        pred = pred_MF - 1 * lams
        return pred

    def l2_norm(self, users, items):
        l2_loss_MF = self.MF.l2_norm(users, items)
        l2_loss = l2_loss_MF + 1000 * torch.sum(self.sigma ** 2)
        return l2_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MetaEmbed,
     lambda: ([], {'dim_1': 4, 'dim_2': 4}),
     lambda: ([], {}),
     True),
    (MetaMF,
     lambda: ([], {'n_user': 4, 'n_item': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     False),
]

class Test_DongHande_AutoDebias(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

