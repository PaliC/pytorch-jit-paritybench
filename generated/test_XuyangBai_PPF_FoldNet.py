import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
dataset = _module
evaluate_3dmatch = _module
evaluate_ppfnet = _module
preparation = _module
utils = _module
input_preparation = _module
chamfer_loss = _module
global_registration = _module
gpu_mem_track = _module
icp_registration = _module
linear_conv1d = _module
model_conv1d = _module
model_linear = _module
fuse_fragments_3DMatch = _module
train = _module
trainer = _module
io = _module

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


import time


import torch


import numpy as np


import numpy


import torch.utils.data as data


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch import nn


import itertools


from torch import optim


import torch.optim as optim


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


class Encoder(nn.Module):

    def __init__(self, num_patches=32, num_points_per_patch=1024):
        super(Encoder, self).__init__()
        self.num_patches = num_patches
        self.num_points_per_patches = num_points_per_patch
        self.conv1 = nn.Linear(4, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(64 + 256, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, input):
        x = self.relu1(self.bn1(self.conv1(input).transpose(-1, -2)))
        local_feature_1 = x.transpose(-1, -2)
        x = self.relu2(self.bn2(self.conv2(x.transpose(-1, -2)).transpose(-1, -2)))
        local_feature_2 = x.transpose(-1, -2)
        x = self.relu3(self.bn3(self.conv3(x.transpose(-1, -2)).transpose(-1, -2)))
        local_feature_3 = x.transpose(-1, -2)
        x = x.transpose(-1, -2)
        x = torch.max(x, 1, keepdim=True)[0]
        global_feature = x.repeat([1, self.num_points_per_patches, 1])
        feature = torch.cat([local_feature_1, global_feature], -1)
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))
        return torch.max(x, 1, keepdim=True)[0]


class Decoder(nn.Module):

    def __init__(self, num_points_per_patch=1024):
        super(Decoder, self).__init__()
        self.m = num_points_per_patch
        self.meshgrid = [[-0.3, 0.3, 32], [-0.3, 0.3, 32]]
        self.mlp1 = nn.Sequential(nn.Linear(514, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
        self.mlp2 = nn.Sequential(nn.Linear(516, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        grid = np.array(list(itertools.product(x, y)))
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
        grid = torch.tensor(grid)
        return grid.float()

    def forward(self, input):
        input = input.repeat(1, self.m, 1)
        grid = self.build_grid(input.shape[0])
        if torch.cuda.is_available():
            grid = grid
        concate1 = torch.cat((input, grid), dim=-1)
        after_folding1 = self.mlp1(concate1)
        concate2 = torch.cat((input, after_folding1), dim=-1)
        after_folding2 = self.mlp2(concate2)
        return after_folding2


class PPFFoldNet(nn.Module):
    """
    This model is similar with PPFFoldNet defined in model.py, the difference is:
        1. I use Linear layer to replace Conv1d because the test shows that Linear is faster.
        2. Different skip connection scheme.
    """

    def __init__(self, num_patches=32, num_points_per_patch=1024):
        super(PPFFoldNet, self).__init__()
        self.encoder = Encoder(num_patches=num_patches, num_points_per_patch=num_points_per_patch)
        self.decoder = Decoder(num_points_per_patch=num_points_per_patch)
        self.loss = ChamferLoss()
        if torch.cuda.is_available():
            summary(self, (1024, 4), batch_size=num_patches)
        else:
            summary(self, (1024, 4), batch_size=num_patches)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        codeword = self.encoder(input.float())
        output = self.decoder(codeword)
        return output

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        return self.loss(input, output)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChamferLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_XuyangBai_PPF_FoldNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

