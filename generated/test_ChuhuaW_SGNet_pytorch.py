import sys
_module = sys.modules[__name__]
del sys
configs = _module
base_configs = _module
ethucy = _module
jaad = _module
pie = _module
JAAD_origin = _module
PIE_origin = _module
dataloaders = _module
datasets = _module
ethucy_data_layer = _module
jaad_data_layer = _module
pie_data_layer = _module
trajectron = _module
losses = _module
cvae = _module
rmse = _module
SGNet = _module
SGNet_CVAE = _module
models = _module
bitrap_np = _module
feature_extractor = _module
utils = _module
data_utils = _module
ethucy_train_utils = _module
ethucy_train_utils_cvae = _module
eval_utils = _module
hevi_train_utils = _module
jaadpie_train_utils_cvae = _module
eval_cvae = _module
eval_deterministic = _module
train_cvae = _module
train_deterministic = _module
eval_cvae = _module
train_cvae = _module
eval_cvae = _module
train_cvae = _module

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


import torch


from torch.utils import data


import random


from copy import deepcopy


import torch.nn as nn


import torch.nn.functional as F


import copy


from collections import defaultdict


from torch import nn


from torch import optim


from torch.nn import functional as F


import torch.nn.utils.rnn as rnn


from torch.distributions import Normal


from torchvision import datasets


from torchvision import transforms


from torchvision import models


import torch.utils.data as data


import time


class rmse_loss(nn.Module):
    """
    Params:
        x_pred: (batch_size, enc_steps, dec_steps, pred_dim)
        x_true: (batch_size, enc_steps, dec_steps, pred_dim)
    Returns:
        rmse: scalar, rmse = \\sum_{i=1:batch_size}()
    """

    def __init__(self):
        super(rmse_loss, self).__init__()

    def forward(self, x_pred, x_true):
        L2_diff = torch.sqrt(torch.sum((x_pred - x_true) ** 2, dim=3))
        L2_all_pred = torch.sum(L2_diff, dim=2)
        L2_mean_pred = torch.mean(L2_all_pred, dim=1)
        L2_mean_pred = torch.mean(L2_mean_pred, dim=0)
        return L2_mean_pred


class ETHUCYFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(ETHUCYFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.embed = nn.Sequential(nn.Linear(6, self.embbed_size), nn.ReLU())

    def forward(self, inputs):
        box_input = inputs
        embedded_box_input = self.embed(box_input)
        return embedded_box_input


class JAADFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(JAADFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size), nn.ReLU())

    def forward(self, inputs):
        box_input = inputs
        embedded_box_input = self.box_embed(box_input)
        return embedded_box_input


class PIEFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(PIEFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size), nn.ReLU())

    def forward(self, inputs):
        box_input = inputs
        embedded_box_input = self.box_embed(box_input)
        return embedded_box_input


_FEATURE_EXTRACTORS = {'PIE': PIEFeatureExtractor, 'JAAD': JAADFeatureExtractor, 'ETH': ETHUCYFeatureExtractor, 'HOTEL': ETHUCYFeatureExtractor, 'UNIV': ETHUCYFeatureExtractor, 'ZARA1': ETHUCYFeatureExtractor, 'ZARA2': ETHUCYFeatureExtractor}


def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)


class SGNet(nn.Module):

    def __init__(self, args):
        super(SGNet, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = build_feature_extractor(args)
        if self.dataset in ['JAAD', 'PIE']:
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.pred_dim), nn.Tanh())
            self.flow_enc_cell = nn.GRUCell(self.hidden_size * 2, self.hidden_size)
        elif self.dataset in ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']:
            self.pred_dim = 2
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.pred_dim))
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size // 4, 1), nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size // 4, 1), nn.ReLU(inplace=True))
        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size // 4), nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True))
        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size // 4, self.hidden_size), nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size // 4, self.hidden_size // 4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4, self.hidden_size)

    def SGE(self, goal_hidden):
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size // 4))
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            goal_traj[:, dec_step, :] = self.regressor(goal_traj_hidden)
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list], dim=1)
        enc_attn = self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim=1).unsqueeze(1)
        goal_for_enc = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def decoder(self, dec_hidden, goal_for_dec):
        dec_traj = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.pred_dim)
        for dec_step in range(self.dec_steps):
            goal_dec_input = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.hidden_size // 4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:], dim=1)
            goal_dec_input[:, dec_step:, :] = goal_dec_input_temp
            dec_attn = self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim=1).unsqueeze(1)
            goal_dec_input = torch.bmm(dec_attn, goal_dec_input).squeeze(1)
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input, dec_dec_input), dim=-1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            dec_traj[:, dec_step, :] = self.regressor(dec_hidden)
        return dec_traj

    def encoder(self, traj_input, flow_input=None, start_index=0):
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size // 4))
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        for enc_step in range(start_index, self.enc_steps):
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:, enc_step, :], goal_for_enc), 1)), traj_enc_hidden)
            if self.dataset in ['JAAD', 'PIE', 'ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']:
                enc_hidden = traj_enc_hidden
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)
            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            dec_traj = self.decoder(dec_hidden, goal_for_dec)
            all_goal_traj[:, enc_step, :, :] = goal_traj
            all_dec_traj[:, enc_step, :, :] = dec_traj
        return all_goal_traj, all_dec_traj

    def forward(self, inputs, start_index=0):
        if self.dataset in ['JAAD', 'PIE']:
            traj_input = self.feature_extractor(inputs)
            all_goal_traj, all_dec_traj = self.encoder(traj_input)
            return all_goal_traj, all_dec_traj
        elif self.dataset in ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']:
            traj_input_temp = self.feature_extractor(inputs[:, start_index:, :])
            traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))
            traj_input[:, start_index:, :] = traj_input_temp
            all_goal_traj, all_dec_traj = self.encoder(traj_input, None, start_index)
            return all_goal_traj, all_dec_traj


def reconstructed_probability(x):
    recon_dist = Normal(0, 1)
    p = recon_dist.log_prob(x).exp().mean(dim=-1)
    return p


class BiTraPNP(nn.Module):

    def __init__(self, args):
        super(BiTraPNP, self).__init__()
        self.args = copy.deepcopy(args)
        self.param_scheduler = None
        self.input_dim = self.args.input_dim
        self.pred_dim = self.args.pred_dim
        self.hidden_size = self.args.hidden_size
        self.nu = args.nu
        self.sigma = args.sigma
        self.node_future_encoder_h = nn.Sequential(nn.Linear(self.input_dim, self.hidden_size // 2), nn.ReLU())
        self.gt_goal_encoder = nn.GRU(input_size=self.pred_dim, hidden_size=self.hidden_size // 2, bidirectional=True, batch_first=True)
        self.p_z_x = nn.Sequential(nn.Linear(self.hidden_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, self.args.LATENT_DIM * 2))
        self.q_z_xy = nn.Sequential(nn.Linear(self.hidden_size + self.hidden_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, self.args.LATENT_DIM * 2))

    def gaussian_latent_net(self, enc_h, cur_state, K, target=None, z_mode=None):
        z_mu_logvar_p = self.p_z_x(enc_h)
        z_mu_p = z_mu_logvar_p[:, :self.args.LATENT_DIM]
        z_logvar_p = z_mu_logvar_p[:, self.args.LATENT_DIM:]
        if target is not None:
            initial_h = self.node_future_encoder_h(cur_state)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            self.gt_goal_encoder.flatten_parameters()
            _, target_h = self.gt_goal_encoder(target, initial_h)
            target_h = target_h.permute(1, 0, 2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.args.LATENT_DIM]
            z_logvar_q = z_mu_logvar_q[:, self.args.LATENT_DIM:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q
            KLD = 0.5 * (z_logvar_q.exp() / z_logvar_p.exp() + (z_mu_p - z_mu_q).pow(2) / z_logvar_p.exp() - 1 + (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = torch.as_tensor(0.0, device=Z_logvar.device)
        with torch.set_grad_enabled(False):
            K_samples = torch.normal(self.nu, self.sigma, size=(enc_h.shape[0], K, self.args.LATENT_DIM))
        probability = reconstructed_probability(K_samples)
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, K, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, K, 1)
        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)
        return Z, KLD, probability

    def forward(self, h_x, last_input, K, target_y=None):
        """
        Params:

        """
        Z, KLD, probability = self.gaussian_latent_net(h_x, last_input, K, target_y, z_mode=False)
        enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        dec_h = enc_h_and_z if self.args.DEC_WITH_Z else h_x
        return dec_h, KLD, probability


class SGNet_CVAE(nn.Module):

    def __init__(self, args):
        super(SGNet_CVAE, self).__init__()
        self.cvae = BiTraPNP(args)
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = build_feature_extractor(args)
        self.pred_dim = args.pred_dim
        self.K = args.K
        self.map = False
        if self.dataset in ['JAAD', 'PIE']:
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.pred_dim), nn.Tanh())
            self.flow_enc_cell = nn.GRUCell(self.hidden_size * 2, self.hidden_size)
        elif self.dataset in ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']:
            self.pred_dim = 2
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.pred_dim))
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size // 4, 1), nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size // 4, 1), nn.ReLU(inplace=True))
        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size // 4), nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size // 4, self.hidden_size), nn.ReLU(inplace=True))
        self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + args.LATENT_DIM, self.hidden_size), nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True))
        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size // 4, self.hidden_size // 4), nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size // 4, self.hidden_size // 4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4, self.hidden_size)

    def SGE(self, goal_hidden):
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size // 4))
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            goal_traj[:, dec_step, :] = self.regressor(goal_traj_hidden)
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list], dim=1)
        enc_attn = self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim=1).unsqueeze(1)
        goal_for_enc = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def cvae_decoder(self, dec_hidden, goal_for_dec):
        batch_size = dec_hidden.size(0)
        K = dec_hidden.shape[1]
        dec_hidden = dec_hidden.view(-1, dec_hidden.shape[-1])
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        for dec_step in range(self.dec_steps):
            goal_dec_input = dec_hidden.new_zeros(batch_size, self.dec_steps, self.hidden_size // 4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:], dim=1)
            goal_dec_input[:, dec_step:, :] = goal_dec_input_temp
            dec_attn = self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim=1).unsqueeze(1)
            goal_dec_input = torch.bmm(dec_attn, goal_dec_input).squeeze(1)
            goal_dec_input = goal_dec_input.unsqueeze(1).repeat(1, K, 1).view(-1, goal_dec_input.shape[-1])
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input, dec_dec_input), dim=-1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            batch_traj = self.regressor(dec_hidden)
            batch_traj = batch_traj.view(-1, K, batch_traj.shape[-1])
            dec_traj[:, dec_step, :, :] = batch_traj
        return dec_traj

    def encoder(self, raw_inputs, raw_targets, traj_input, flow_input=None, start_index=0):
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size // 4))
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        total_probabilities = traj_input.new_zeros((traj_input.size(0), self.enc_steps, self.K))
        total_KLD = 0
        for enc_step in range(start_index, self.enc_steps):
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:, enc_step, :], goal_for_enc), 1)), traj_enc_hidden)
            enc_hidden = traj_enc_hidden
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            all_goal_traj[:, enc_step, :, :] = goal_traj
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)
            if self.training:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:, enc_step, :], self.K, raw_targets[:, enc_step, :, :])
            else:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:, enc_step, :], self.K)
            total_probabilities[:, enc_step, :] = probability
            total_KLD += KLD
            cvae_dec_hidden = self.cvae_to_dec_hidden(cvae_hidden)
            if self.map:
                map_input = flow_input
                cvae_dec_hidden = (cvae_dec_hidden + map_input.unsqueeze(1)) / 2
            all_cvae_dec_traj[:, enc_step, :, :, :] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec)
        return all_goal_traj, all_cvae_dec_traj, total_KLD, total_probabilities

    def forward(self, inputs, map_mask=None, targets=None, start_index=0, training=True):
        self.training = training
        if torch.is_tensor(start_index):
            start_index = start_index[0].item()
        if self.dataset in ['JAAD', 'PIE']:
            traj_input = self.feature_extractor(inputs)
            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input)
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities
        elif self.dataset in ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']:
            traj_input_temp = self.feature_extractor(inputs[:, start_index:, :])
            traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))
            traj_input[:, start_index:, :] = traj_input_temp
            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, None, start_index)
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (JAADFeatureExtractor,
     lambda: ([], {'args': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PIEFeatureExtractor,
     lambda: ([], {'args': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (rmse_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ChuhuaW_SGNet_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

