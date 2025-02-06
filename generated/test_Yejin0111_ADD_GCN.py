import sys
_module = sys.modules[__name__]
del sys
data = _module
coco = _module
test_dataset = _module
voc = _module
main = _module
models = _module
add_gcn = _module
trainer = _module
util = _module

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


import random


import torch


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


from torch.utils.data import Dataset


import numpy as np


import warnings


import torch.backends.cudnn as cudnn


import torch.nn as nn


import time


from torch.autograd import Variable


import math


class DynamicGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()
        self.static_adj = nn.Sequential(nn.Conv1d(num_nodes, num_nodes, 1, bias=False), nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(nn.Conv1d(in_features, out_features, 1), nn.LeakyReLU(0.2))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)
        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x


class ADD_GCN(nn.Module):

    def __init__(self, model, num_classes):
        super(ADD_GCN, self).__init__()
        self.features = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4)
        self.num_classes = num_classes
        self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1, 1), bias=False)
        self.conv_transform = nn.Conv2d(2048, 1024, (1, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape: 
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)
        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward_dgcn(self, x):
        x = self.gcn(x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        out1 = self.forward_classification_sm(x)
        v = self.forward_sam(x)
        z = self.forward_dgcn(v)
        z = v + z
        out2 = self.last_linear(z)
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        return out1, out2

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [{'params': self.features.parameters(), 'lr': lr * lrp}, {'params': large_lr_layers, 'lr': lr}]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DynamicGraphConvolution,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'num_nodes': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_Yejin0111_ADD_GCN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

