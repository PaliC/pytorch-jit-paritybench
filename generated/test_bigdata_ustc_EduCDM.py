import sys
_module = sys.modules[__name__]
del sys
DINA = _module
EM = _module
DINA = _module
GD = _module
FuzzyCDF = _module
modules = _module
ICD = _module
etl = _module
etl = _module
utils = _module
metrics = _module
sym = _module
fit_eval = _module
net = _module
dtn = _module
mirt = _module
ncd = _module
net = _module
pos_linear = _module
DINA = _module
IRT = _module
MIRT = _module
NCDM = _module
IRR = _module
pair_etl = _module
point_etl = _module
loss = _module
IRT = _module
irt = _module
KaNCD = _module
MCD = _module
MIRT = _module
NCDM = _module
EduCDM = _module
meta = _module
DINA = _module
IRT = _module
KaNCD = _module
MCD = _module
MIRT = _module
NCDM = _module
setup = _module
tests = _module
dina = _module
em = _module
conftest = _module
test_dina = _module
gd = _module
conftest = _module
test_gddina = _module
fuzzycdf = _module
test_fuzzycdf = _module
icd = _module
test_mirt = _module
test_ncd = _module
irr = _module
test_irt = _module
test_ncdm = _module
test_emirt = _module
conftest = _module
test_gdirt = _module
kancd = _module
conftest = _module
test_kancd = _module
mcd = _module
conftest = _module
test_mcd = _module
conftest = _module
ncdm = _module
conftest = _module

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


import numpy as np


import torch


from torch import nn


from sklearn.metrics import roc_auc_score


from sklearn.metrics import accuracy_score


import torch.autograd as autograd


import torch.nn.functional as F


import pandas as pd


from copy import deepcopy


from torch import Tensor


from torch import LongTensor


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


import math


from scipy.stats import entropy


import torch.nn as nn


import torch.optim as optim


import random


class DINANet(nn.Module):

    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(DINANet, self).__init__()
        self._user_num = user_num
        self._item_num = item_num
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess
        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

    def forward(self, user, item, knowledge, *args):
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100, 1e-06), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1), dim=1)
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)


class STEFunction(autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):

    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class STEDINANet(DINANet):

    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(STEDINANet, self).__init__(user_num, item_num, hidden_dim, max_slip, max_guess, *args, **kwargs)
        self.sign = StraightThroughEstimator()

    def forward(self, user, item, knowledge, *args):
        theta = self.sign(self.theta(user))
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        mask_theta = (knowledge == 0) + (knowledge == 1) * theta
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)


class DTN(nn.Module):

    def __init__(self, input_dim, know_dim):
        self.know_dim = know_dim
        self.input_dim = input_dim
        self.fea_dim = 64
        super(DTN, self).__init__()
        self.emb = nn.Sequential(nn.Embedding(self.input_dim, self.fea_dim))
        self.feature_net = nn.Sequential(nn.Linear(self.fea_dim, self.know_dim))

    def avg_pool(self, data, mask: 'torch.Tensor'):
        mask_data = mask_sequence(data, mask)
        rs = torch.sum(mask_data.permute(0, 2, 1), dim=-1)
        len_mask = mask.reshape((-1, 1))
        len_mask = len_mask.expand(len_mask.size()[0], self.know_dim)
        rs = torch.div(rs, len_mask)
        return rs

    def forward(self, log, mask):
        emb = self.emb(log)
        fea = self.feature_net(emb)
        trait = self.avg_pool(fea, mask)
        return trait


def irt2pl(theta, a, b, *, F=np):
    """

    Parameters
    ----------
    theta
    a
    b
    F

    Returns
    -------

    Examples
    --------
    >>> theta = [1, 0.5, 0.3]
    >>> a = [-3, 1, 3]
    >>> b = 0.5
    >>> float(irt2pl(theta, a, b)) # doctest: +ELLIPSIS
    0.109...
    >>> theta = [[1, 0.5, 0.3], [2, 1, 0]]
    >>> a = [[-3, 1, 3], [-3, 1, 3]]
    >>> b = [0.5, 0.5]
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    array([0.109..., 0.004...])
    """
    return 1 / (1 + F.exp(-F.sum(F.multiply(a, theta), axis=-1) + b))


class MIRTNet(nn.Module):

    def __init__(self, user_num, item_num, latent_dim, a_range, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)
        self.a_range = a_range

    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        b = torch.squeeze(self.b(item), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(theta, a, b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)


class PosLinear(nn.Linear):

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class NCDMNet(nn.Module):

    def __init__(self, trait_dim, know_dim):
        super(NCDMNet, self).__init__()
        self.knowledge_dim = know_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        self.l_dtn_theta_fc = nn.Linear(trait_dim, self.prednet_input_len)
        self.i_dtn_kd_fc = nn.Linear(trait_dim, self.prednet_input_len)
        self.i_dtn_ed_fc = nn.Linear(trait_dim, self.prednet_input_len)
        self.int_fc = nn.Sequential(PosLinear(self.prednet_input_len, self.prednet_len1), nn.Sigmoid(), nn.Dropout(p=0.5), PosLinear(self.prednet_len1, self.prednet_len2), nn.Sigmoid(), nn.Dropout(p=0.5), PosLinear(self.prednet_len2, 1), nn.Sigmoid())

    def u_theta(self, u_trait):
        return torch.sigmoid(self.l_dtn_theta_fc(u_trait))

    def i_difficulty(self, v_trait):
        return torch.sigmoid(self.i_dtn_kd_fc(v_trait))

    def i_discrimination(self, v_trait):
        return torch.sigmoid(self.i_dtn_ed_fc(v_trait))

    def forward(self, u_trait, v_trait, v_know):
        theta = self.u_theta(u_trait)
        difficulty = self.i_difficulty(v_trait)
        discrimination = self.i_discrimination(v_trait)
        input_x = discrimination * (theta - difficulty) * v_know
        output_1 = self.int_fc(input_x)
        return output_1.view(-1), theta, discrimination, difficulty

    def int_f(self, theta, a, b, know):
        return self.int_fc(a * (theta - b) * know).view(-1)


class ICD(nn.Module):

    def __init__(self, user_n, item_n, know_n, cdm='ncd'):
        super(ICD, self).__init__()
        self.l_dtn = DTN(2 * item_n + 1, know_n)
        self.i_dtn = DTN(2 * user_n + 1, know_n)
        self.cdm_name = cdm
        if cdm == 'ncd':
            self.cdm = NCDMNet(know_n, know_n)
        elif cdm == 'mirt':
            self.cdm = MIRTNet(know_n)
        else:
            raise ValueError()
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, u2i, u_mask, i2u, i_mask, i2k):
        u_trait = self.l_dtn(u2i, u_mask)
        v_trait = self.i_dtn(i2u, i_mask)
        return self.cdm(u_trait, v_trait, i2k)

    def get_user_profiles(self, batches):
        device = next(self.parameters()).device
        ids = []
        traits = []
        for _id, records, r_mask in tqdm(batches, 'getting user profiles'):
            ids.append(_id)
            traits.append(self.cdm.u_theta(self.l_dtn(records.to(device), r_mask.to(device))))
        obj = {'uid': torch.cat(ids), 'u_trait': torch.cat(traits)}
        return obj

    def get_item_profiles(self, batches):
        device = next(self.parameters()).device
        ids = []
        a = []
        b = []
        for _id, records, r_mask in tqdm(batches, 'getting item profiles'):
            v_trait = self.i_dtn(records, r_mask)
            ids.append(_id.cpu())
            a.append(self.cdm.i_discrimination(v_trait))
            b.append(self.cdm.i_difficulty(v_trait))
        obj = {'iid': torch.cat(ids), 'ia': torch.cat(a), 'ib': torch.cat(b)}
        return obj


class DualICD(nn.Module):

    def __init__(self, stat_net: 'ICD', net: 'ICD', alpha=0.999):
        super(DualICD, self).__init__()
        self.stat_net = stat_net
        self.net = net
        self.alpha = alpha

    def momentum_weight_update(self, pre_net, train_select=None):
        """
        Momentum update of ICD
        """
        pre_net_params = collect_params(pre_net, train_select)
        net_params = collect_params(self.net, train_select)
        for param_pre, param_now in zip(pre_net_params, net_params):
            param_now.data = param_pre.data * self.alpha + param_now.data * (1.0 - self.alpha)

    def forward(self, u2i, u_mask, i2u, i_mask, i2k):
        output, theta, a, b = self.net(u2i, u_mask, i2u, i_mask, i2k)
        _, stat_theta, stat_a, stat_b = self.stat_net(u2i, u_mask, i2u, i_mask, i2k)
        return output, theta, a, b, stat_theta, stat_a, stat_b


class EmbICD(nn.Module):

    def __init__(self, int_fc, weights):
        super(EmbICD, self).__init__()
        self.theta_emb = nn.Embedding(*weights[0].size(), _weight=weights[0])
        self.a_emb = nn.Embedding(*weights[1].size(), _weight=weights[1])
        if len(weights[2].size()) == 1:
            self.b_emb = nn.Embedding(weights[2].size()[0], 1, _weight=torch.unsqueeze(weights[2], 1))
        else:
            self.b_emb = nn.Embedding(*weights[2].size(), _weight=weights[2])
        self.int_fc = int_fc
        self._user_id2idx = {}
        self._item_id2idx = {}

    def build_user_id2idx(self, users):
        idx = 0
        for user_id in users:
            if user_id not in self._user_id2idx:
                self._user_id2idx[user_id] = idx
                idx += 1

    def build_item_id2idx(self, items):
        idx = 0
        for item_id in items:
            if item_id not in self._item_id2idx:
                self._item_id2idx[item_id] = idx
                idx += 1

    def user_id2idx(self, users):
        users_idx = []
        for user in users:
            users_idx.append(self._user_id2idx[user])
        return users_idx

    def item_id2idx(self, items):
        items_idx = []
        for item in items:
            items_idx.append(self._item_id2idx[item])
        return items_idx

    def forward(self, user_idx, item_idx, know):
        theta = self.theta_emb(user_idx).detach()
        a = self.a_emb(item_idx).detach()
        b = self.b_emb(item_idx).detach()
        theta.requires_grad_(True)
        a.requires_grad_(True)
        b.requires_grad_(True)
        return self.int_fc(theta, a, torch.squeeze(b), know).view(-1), theta, a, b


class DeltaTraitLoss(nn.Module):

    def __init__(self):
        super(DeltaTraitLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, theta, a, b, stat_theta, stat_a, stat_b):
        return self.mse_loss(theta, stat_theta) + self.mse_loss(a, stat_a) + self.mse_loss(b, stat_b)


class DualLoss(nn.Module):

    def __init__(self, beta=0.95, *args, **kwargs):
        super(DualLoss, self).__init__()
        self.beta = beta
        self.bce = nn.BCELoss(*args, **kwargs)
        self.delta_trait = DeltaTraitLoss()

    def forward(self, pred, truth, theta, a, b, stat_theta, stat_a, stat_b):
        return self.beta * self.bce(pred, truth) + (1.0 - self.beta) * self.delta_trait(theta, a, b, stat_theta, stat_a, stat_b)


class PairSCELoss(nn.Module):

    def __init__(self):
        super(PairSCELoss, self).__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, pred1, pred2, sign=1, *args):
        """
        sign is either 1 or -1
        could be seen as predicting the sign based on the pred1 and pred2
        1: pred1 should be greater than pred2
        -1: otherwise
        """
        pred = torch.stack([pred1, pred2], dim=1)
        return self._loss(pred, ((torch.ones(pred1.shape[0], device=pred.device) - sign) / 2).long())


def irf(theta, a, b, c, D=1.702, *, F=np):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))


irt3pl = irf


class IRTNet(nn.Module):

    def __init__(self, user_num, item_num, value_range, a_range, irf_kwargs=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        self.value_range = value_range
        self.a_range = a_range

    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        c = torch.sigmoid(c)
        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
        return self.irf(theta, a, b, c, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        super(Net, self).__init__()
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        return output_1.view(-1)


class MFNet(nn.Module):
    """Matrix Factorization Network"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MFNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        self.response = nn.Linear(2 * self.latent_dim, 1)

    def forward(self, user_id, item_id):
        user = self.user_embedding(user_id)
        item = self.item_embedding(item_id)
        return torch.squeeze(torch.sigmoid(self.response(torch.cat([user, item], dim=-1))), dim=-1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DeltaTraitLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DualLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NCDMNet,
     lambda: ([], {'trait_dim': 4, 'know_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PosLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StraightThroughEstimator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_bigdata_ustc_EduCDM(_paritybench_base):
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

