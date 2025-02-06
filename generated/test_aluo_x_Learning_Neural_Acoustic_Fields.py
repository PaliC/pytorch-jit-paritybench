import sys
_module = sys.modules[__name__]
del sys
master = _module
make_data_aac = _module
make_data_opus = _module
sound_loader = _module
metadata = _module
modules = _module
networks = _module
options = _module
results = _module
cache_feature_NAF = _module
cache_test_NAF = _module
cache_test_baseline = _module
compute_T60_err_NAF = _module
compute_T60_err_baseline = _module
compute_spectral_NAF = _module
compute_spectral_baseline = _module
lin_probe_NAF = _module
test_utils = _module
vis_feat_NAF = _module
vis_loudness_NAF = _module
train = _module
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


import numpy.random


import torch


import numpy as np


import random


from torch import nn


import math


import matplotlib.pyplot as plt


from inspect import getsourcefile


import string


from sklearn.metrics import explained_variance_score


from scipy.io import wavfile


from sklearn.manifold import TSNE


import torch.multiprocessing as mp


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from time import time


import functools


class embedding_module_log(nn.Module):

    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0 ** torch.from_numpy(np.linspace(start=0.0, stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input * freq))
        return torch.cat(out_list, dim=self.ch_dim)


class basic_project2(nn.Module):

    def __init__(self, input_ch, output_ch):
        super(basic_project2, self).__init__()
        self.proj = nn.Linear(input_ch, output_ch, bias=True)

    def forward(self, x):
        return self.proj(x)


class kernel_linear_act(nn.Module):

    def __init__(self, input_ch, output_ch):
        super(kernel_linear_act, self).__init__()
        self.block = nn.Sequential(nn.LeakyReLU(negative_slope=0.1), basic_project2(input_ch, output_ch))

    def forward(self, input_x):
        return self.block(input_x)


def distance(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res


def fit_predict_torch(input_pos: 'torch.Tensor', input_target: 'torch.Tensor', predict_pos: 'torch.Tensor', bandwidth: 'torch.Tensor') ->torch.Tensor:
    dist_vector = -distance(predict_pos, input_pos)
    gauss_dist = torch.exp(dist_vector / (2.0 * torch.square(bandwidth.unsqueeze(0))))
    magnitude = torch.sum(gauss_dist, dim=1, keepdim=True)
    out = torch.mm(gauss_dist, input_target) / magnitude
    return out


class kernel_residual_fc_embeds(nn.Module):

    def __init__(self, input_ch, intermediate_ch=512, grid_ch=64, num_block=8, output_ch=1, grid_gap=0.25, grid_bandwidth=0.25, bandwidth_min=0.1, bandwidth_max=0.5, float_amt=0.1, min_xy=None, max_xy=None, probe=False):
        super(kernel_residual_fc_embeds, self).__init__()
        for k in range(num_block - 1):
            self.register_parameter('left_right_{}'.format(k), nn.Parameter(torch.randn(1, 1, 2, intermediate_ch) / math.sqrt(intermediate_ch), requires_grad=True))
        for k in range(4):
            self.register_parameter('rot_{}'.format(k), nn.Parameter(torch.randn(num_block - 1, 1, 1, intermediate_ch) / math.sqrt(intermediate_ch), requires_grad=True))
        self.proj = basic_project2(input_ch + int(2 * grid_ch), intermediate_ch)
        self.residual_1 = nn.Sequential(basic_project2(input_ch + 128, intermediate_ch), nn.LeakyReLU(negative_slope=0.1), basic_project2(intermediate_ch, intermediate_ch))
        self.layers = torch.nn.ModuleList()
        for k in range(num_block - 2):
            self.layers.append(kernel_linear_act(intermediate_ch, intermediate_ch))
        self.out_layer = nn.Linear(intermediate_ch, output_ch)
        self.blocks = len(self.layers)
        self.probe = probe
        grid_coors_x = np.arange(min_xy[0], max_xy[0], grid_gap)
        grid_coors_y = np.arange(min_xy[1], max_xy[1], grid_gap)
        grid_coors_x, grid_coors_y = np.meshgrid(grid_coors_x, grid_coors_y)
        grid_coors_x = grid_coors_x.flatten()
        grid_coors_y = grid_coors_y.flatten()
        xy_train = np.array([grid_coors_x, grid_coors_y]).T
        self.bandwidth_min = bandwidth_min
        self.bandwidth_max = bandwidth_max
        self.float_amt = float_amt
        self.bandwidths = nn.Parameter(torch.zeros(len(grid_coors_x)) + grid_bandwidth, requires_grad=True)
        self.register_buffer('grid_coors_xy', torch.from_numpy(xy_train).float(), persistent=True)
        self.xy_offset = nn.Parameter(torch.zeros_like(self.grid_coors_xy), requires_grad=True)
        self.grid_0 = nn.Parameter(torch.randn(len(grid_coors_x), grid_ch, device='cpu').float() / np.sqrt(float(grid_ch)), requires_grad=True)

    def forward(self, input_stuff, rot_idx, sound_loc=None):
        SAMPLES = input_stuff.shape[1]
        sound_loc_v0 = sound_loc[..., :2]
        sound_loc_v1 = sound_loc[..., 2:]
        self.bandwidths.data = torch.clamp(self.bandwidths.data, self.bandwidth_min, self.bandwidth_max)
        grid_coors_baseline = self.grid_coors_xy + torch.tanh(self.xy_offset) * self.float_amt
        grid_feat_v0 = fit_predict_torch(grid_coors_baseline, self.grid_0, sound_loc_v0, self.bandwidths)
        grid_feat_v1 = fit_predict_torch(grid_coors_baseline, self.grid_0, sound_loc_v1, self.bandwidths)
        total_grid = torch.cat((grid_feat_v0, grid_feat_v1), dim=-1).unsqueeze(1).expand(-1, SAMPLES, -1)
        my_input = torch.cat((total_grid, input_stuff), dim=-1)
        rot_latent = torch.stack([getattr(self, 'rot_{}'.format(rot_idx_single)) for rot_idx_single in rot_idx], dim=0)
        out = self.proj(my_input).unsqueeze(2).repeat(1, 1, 2, 1) + getattr(self, 'left_right_0') + rot_latent[:, 0]
        for k in range(len(self.layers)):
            out = self.layers[k](out) + getattr(self, 'left_right_{}'.format(k + 1)) + rot_latent[:, k + 1]
            if k == self.blocks // 2 - 1:
                out = out + self.residual_1(my_input).unsqueeze(2).repeat(1, 1, 2, 1)
        if self.probe:
            return out
        return self.out_layer(out)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (basic_project2,
     lambda: ([], {'input_ch': 4, 'output_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (embedding_module_log,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (kernel_linear_act,
     lambda: ([], {'input_ch': 4, 'output_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_aluo_x_Learning_Neural_Acoustic_Fields(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

