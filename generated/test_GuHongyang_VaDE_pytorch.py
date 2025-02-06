import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
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


import torch


from torchvision.datasets import MNIST


from torchvision.transforms import ToTensor


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import numpy as np


from torch.optim import Adam


from sklearn.metrics import accuracy_score


from torch.optim.lr_scheduler import StepLR


from sklearn.manifold import TSNE


import torch.nn as nn


import torch.nn.functional as F


import itertools


from sklearn.mixture import GaussianMixture


def block(in_c, out_c):
    layers = [nn.Linear(in_c, out_c), nn.ReLU(True)]
    return layers


class Encoder(nn.Module):

    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(*block(input_dim, inter_dims[0]), *block(inter_dims[0], inter_dims[1]), *block(inter_dims[1], inter_dims[2]))
        self.mu_l = nn.Linear(inter_dims[-1], hid_dim)
        self.log_sigma2_l = nn.Linear(inter_dims[-1], hid_dim)

    def forward(self, x):
        e = self.encoder(x)
        mu = self.mu_l(e)
        log_sigma2 = self.log_sigma2_l(e)
        return mu, log_sigma2


class Decoder(nn.Module):

    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(*block(hid_dim, inter_dims[-1]), *block(inter_dims[-1], inter_dims[-2]), *block(inter_dims[-2], inter_dims[-3]), nn.Linear(inter_dims[-3], input_dim), nn.Sigmoid())

    def forward(self, z):
        x_pro = self.decoder(z)
        return x_pro


class VaDE(nn.Module):

    def __init__(self, args):
        super(VaDE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.pi_ = nn.Parameter(torch.FloatTensor(args.nClusters).fill_(1) / args.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)
        self.args = args

    def pre_train(self, dataloader, pre_epoch=10):
        if not os.path.exists('./pretrain_model.pk'):
            Loss = nn.MSELoss()
            opti = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()))
            None
            epoch_bar = tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L = 0
                for x, y in dataloader:
                    if self.args.cuda:
                        x = x
                    z, _ = self.encoder(x)
                    x_ = self.decoder(z)
                    loss = Loss(x, x_)
                    L += loss.detach().cpu().numpy()
                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                epoch_bar.write('L2={:.4f}'.format(L / len(dataloader)))
            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())
            Z = []
            Y = []
            with torch.no_grad():
                for x, y in dataloader:
                    if self.args.cuda:
                        x = x
                    z1, z2 = self.encoder(x)
                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)
                    Y.append(y)
            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()
            gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')
            pre = gmm.fit_predict(Z)
            None
            self.pi_.data = torch.from_numpy(gmm.weights_).float()
            self.mu_c.data = torch.from_numpy(gmm.means_).float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).float())
            torch.save(self.state_dict(), './pretrain_model.pk')
        else:
            self.load_state_dict(torch.load('./pretrain_model.pk'))

    def predict(self, x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))
        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def ELBO_Loss(self, x, L=1):
        det = 1e-10
        L_rec = 0
        z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
            x_pro = self.decoder(z)
            L_rec += F.binary_cross_entropy(x_pro, x)
        L_rec /= L
        Loss = L_rec * x.size(1)
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det
        yita_c = yita_c / yita_c.sum(1).view(-1, 1)
        Loss += 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) + torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) + (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))
        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / yita_c), 1)) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))
        return Loss

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.args.nClusters):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1)

