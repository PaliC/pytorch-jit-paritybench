import sys
_module = sys.modules[__name__]
del sys
models = _module
mlp = _module
motion_pred = _module
rnn = _module
eval = _module
exp_dlow = _module
exp_vae = _module
config = _module
dataset = _module
dataset_h36m = _module
dataset_humaneva = _module
skeleton = _module
visualization = _module
utils = _module
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


import torch.nn as nn


import torch


import numpy as np


from torch import nn


from torch.nn import functional as F


from scipy.spatial.distance import pdist


from scipy.spatial.distance import squareform


import math


import time


from torch import optim


from torch.utils.tensorboard import SummaryWriter


from torch.optim import lr_scheduler


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


def batch_to(dst, *args):
    return [(x if x is not None else None) for x in args]


zeros = torch.zeros


class RNN(nn.Module):

    def __init__(self, input_dim, out_dim, cell_type='lstm', bi_dir=False):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cell_type = cell_type
        self.bi_dir = bi_dir
        self.mode = 'batch'
        rnn_cls = nn.LSTMCell if cell_type == 'lstm' else nn.GRUCell
        hidden_dim = out_dim // 2 if bi_dir else out_dim
        self.rnn_f = rnn_cls(self.input_dim, hidden_dim)
        if bi_dir:
            self.rnn_b = rnn_cls(self.input_dim, hidden_dim)
        self.hx, self.cx = None, None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, batch_size=1, hx=None, cx=None):
        if self.mode == 'step':
            self.hx = zeros((batch_size, self.rnn_f.hidden_size)) if hx is None else hx
            if self.cell_type == 'lstm':
                self.cx = zeros((batch_size, self.rnn_f.hidden_size)) if cx is None else cx

    def forward(self, x):
        if self.mode == 'step':
            self.hx, self.cx = batch_to(x.device, self.hx, self.cx)
            if self.cell_type == 'lstm':
                self.hx, self.cx = self.rnn_f(x, (self.hx, self.cx))
            else:
                self.hx = self.rnn_f(x, self.hx)
            rnn_out = self.hx
        else:
            rnn_out_f = self.batch_forward(x)
            if not self.bi_dir:
                return rnn_out_f
            rnn_out_b = self.batch_forward(x, reverse=True)
            rnn_out = torch.cat((rnn_out_f, rnn_out_b), 2)
        return rnn_out

    def batch_forward(self, x, reverse=False):
        rnn = self.rnn_b if reverse else self.rnn_f
        rnn_out = []
        hx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        if self.cell_type == 'lstm':
            cx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        ind = reversed(range(x.size(0))) if reverse else range(x.size(0))
        for t in ind:
            if self.cell_type == 'lstm':
                hx, cx = rnn(x[t, ...], (hx, cx))
            else:
                hx = rnn(x[t, ...], hx)
            rnn_out.append(hx.unsqueeze(0))
        if reverse:
            rnn_out.reverse()
        rnn_out = torch.cat(rnn_out, 0)
        return rnn_out


class VAE(nn.Module):

    def __init__(self, nx, ny, nz, horizon, specs):
        super(VAE, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', True)
        self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(2 * nh_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(ny + nz + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y):
        if self.e_birnn:
            h_y = self.e_rnn(y).mean(dim=0)
        else:
            h_y = self.e_rnn(y)[-1]
        return h_y

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x, h_y), dim=1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        h_x = self.encode_x(x)
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])
        y = []
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            rnn_in = torch.cat([h_x, z, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(x, z), mu, logvar

    def sample_prior(self, x):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z)


ones = torch.ones


class NFDiag(nn.Module):

    def __init__(self, nx, ny, nk, specs):
        super(NFDiag, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nk = nk
        self.nh = nh = specs.get('nh_mlp', [300, 200])
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.fix_first = fix_first = specs.get('fix_first', False)
        self.nac = nac = nk - 1 if fix_first else nk
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.mlp = MLP(nh_rnn, nh)
        self.head_A = nn.Linear(nh[-1], ny * nac)
        self.head_b = nn.Linear(nh[-1], ny * nac)

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode(self, x, y):
        if self.fix_first:
            z = y
        else:
            h_x = self.encode_x(x)
            h = self.mlp(h_x)
            a = self.head_A(h).view(-1, self.nk, self.ny)[:, 0, :]
            b = self.head_b(h).view(-1, self.nk, self.ny)[:, 0, :]
            z = (y - b) / a
        return z

    def forward(self, x, z=None):
        h_x = self.encode_x(x)
        if z is None:
            z = torch.randn((h_x.shape[0], self.ny), device=x.device)
        z = z.repeat_interleave(self.nk, dim=0)
        h = self.mlp(h_x)
        if self.fix_first:
            a = self.head_A(h).view(-1, self.nac, self.ny)
            b = self.head_b(h).view(-1, self.nac, self.ny)
            a = torch.cat((ones(h_x.shape[0], 1, self.ny, device=x.device), a), dim=1).view(-1, self.ny)
            b = torch.cat((zeros(h_x.shape[0], 1, self.ny, device=x.device), b), dim=1).view(-1, self.ny)
        else:
            a = self.head_A(h).view(-1, self.ny)
            b = self.head_b(h).view(-1, self.ny)
        y = a * z + b
        return y, a, b

    def sample(self, x, z=None):
        return self.forward(x, z)[0]

    def get_kl(self, a, b):
        var = a ** 2
        KLD = -0.5 * torch.sum(1 + var.log() - b.pow(2) - var)
        return KLD


class NFFull(nn.Module):

    def __init__(self, nx, ny, nk, specs):
        super(NFFull, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nk = nk
        self.nh = nh = specs.get('nh_mlp', [300, 200])
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.fix_first = fix_first = specs.get('fix_first', False)
        self.nac = nac = nk - 1 if fix_first else nk
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.mlp = MLP(nh_rnn, nh)
        self.head_A = nn.Linear(nh[-1], ny * ny * nac)
        self.head_b = nn.Linear(nh[-1], ny * nac)

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode(self, x, y):
        z = y
        return z

    def forward(self, x, z=None):
        h_x = self.encode_x(x)
        if z is None:
            z = torch.randn((h_x.shape[0], self.ny, 1), device=x.device)
        else:
            z = z.unsqueeze(2)
        z = z.repeat_interleave(self.nk, dim=0)
        h = self.mlp(h_x)
        if self.fix_first:
            A = self.head_A(h).view(-1, self.nac, self.ny, self.ny)
            b = self.head_b(h).view(-1, self.nac, self.ny)
            cA = torch.eye(self.ny, device=x.device).repeat((h_x.shape[0], 1, 1, 1))
            A = torch.cat((cA, A), dim=1).view(-1, self.ny, self.ny)
            b = torch.cat((zeros(h_x.shape[0], 1, self.ny, device=x.device), b), dim=1).view(-1, self.ny)
        else:
            A = self.head_A(h).view(-1, self.ny, self.ny)
            b = self.head_b(h).view(-1, self.ny)
        y = A.bmm(z).squeeze(-1) + b
        return y, A, b

    def sample(self, x, z=None):
        return self.forward(x, z)[0]

    def get_kl(self, A, b):
        var = A.bmm(A.transpose(1, 2))
        KLD = -0.5 * (A.shape[-1] + torch.log(torch.det(var)) - b.pow(2).sum(dim=1) - (A * A).sum(dim=-1).sum(dim=-1))
        return KLD.sum()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RNN,
     lambda: ([], {'input_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_Khrylx_DLow(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

