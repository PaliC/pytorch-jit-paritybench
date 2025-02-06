import sys
_module = sys.modules[__name__]
del sys
master = _module
idec = _module
utils = _module

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


from sklearn.cluster import KMeans


from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


from sklearn.metrics import adjusted_rand_score as ari_score


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from torch.optim import Adam


from torch.utils.data import DataLoader


from torch.nn import Linear


from torch.utils.data import Dataset


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar, z


def pretrain_ae(model):
    """
    pretrain autoencoder
    """
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    None
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(200):
        total_loss = 0.0
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        None
        torch.save(model.state_dict(), args.pretrain_path)
    None


class IDEC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, alpha=1, pretrain_path='data/ae_mnist.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path
        self.ae = AE(n_enc_1=n_enc_1, n_enc_2=n_enc_2, n_enc_3=n_enc_3, n_dec_1=n_dec_1, n_dec_2=n_dec_2, n_dec_3=n_dec_3, n_input=n_input, n_z=n_z)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        None

    def forward(self, x):
        x_bar, z = self.ae(x)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AE,
     lambda: ([], {'n_enc_1': 4, 'n_enc_2': 4, 'n_enc_3': 4, 'n_dec_1': 4, 'n_dec_2': 4, 'n_dec_3': 4, 'n_input': 4, 'n_z': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IDEC,
     lambda: ([], {'n_enc_1': 4, 'n_enc_2': 4, 'n_enc_3': 4, 'n_dec_1': 4, 'n_dec_2': 4, 'n_dec_3': 4, 'n_input': 4, 'n_z': 4, 'n_clusters': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_dawnranger_IDEC_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

