import sys
_module = sys.modules[__name__]
del sys
src = _module
graph_net = _module
main_cgct = _module
main_dcgct = _module
networks = _module
preprocess = _module
trainer = _module
transfer_loss = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


import random


import numpy as np


from torch.utils.data import DataLoader


import math


from torchvision import models


import warnings


from torch.utils.data import Dataset


from torchvision import transforms


class NodeNet(nn.Module):

    def __init__(self, in_features, num_features, device, ratio=(2, 1)):
        super(NodeNet, self).__init__()
        num_features_list = [(num_features * r) for r in ratio]
        self.device = device
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=num_features_list[l - 1] if l > 0 else in_features * 2, out_channels=num_features_list[l], kernel_size=1, bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            if l < len(num_features_list) - 1:
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1)
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        aggr_feat = torch.bmm(edge_feat.squeeze(1), node_feat)
        node_feat = torch.cat([node_feat, aggr_feat], -1).transpose(1, 2)
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2)
        node_feat = node_feat.squeeze(-1).squeeze(0)
        return node_feat


class EdgeNet(nn.Module):

    def __init__(self, in_features, num_features, device, ratio=(2, 1)):
        super(EdgeNet, self).__init__()
        num_features_list = [(num_features * r) for r in ratio]
        self.device = device
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=num_features_list[l - 1] if l > 0 else in_features, out_channels=num_features_list[l], kernel_size=1, bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        layer_list['conv_out'] = nn.Conv2d(in_channels=num_features_list[-1], out_channels=1, kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_feat):
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze(1).squeeze(0)
        force_edge_feat = torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1)
        edge_feat = sim_val + force_edge_feat
        edge_feat = edge_feat + 1e-06
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1)
        return edge_feat, sim_val


class ClassifierGNN(nn.Module):

    def __init__(self, in_features, edge_features, nclasses, device):
        super(ClassifierGNN, self).__init__()
        self.edge_net = EdgeNet(in_features=in_features, num_features=edge_features, device=device)
        self.node_net = NodeNet(in_features=in_features, num_features=nclasses, device=device)
        self.mask_val = -1

    def label2edge(self, targets):
        """ convert node labels to affinity mask for backprop"""
        num_sample = targets.size()[1]
        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float()
        target_edge_mask = (torch.eq(label_i, self.mask_val) + torch.eq(label_j, self.mask_val)).type(torch.bool)
        source_edge_mask = ~target_edge_mask
        init_edge = edge * source_edge_mask.float()
        return init_edge[0], source_edge_mask

    def forward(self, init_node_feat):
        edge_feat, edge_sim = self.edge_net(init_node_feat)
        logits_gnn = self.node_net(init_node_feat, edge_feat)
        return logits_gnn, edge_sim


class RandomLayer(nn.Module):

    def __init__(self, input_dim_list, output_dim, device):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class AdversarialNetwork(nn.Module):

    def __init__(self, in_feature, hidden_size, ndomains):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, ndomains)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.ndomains = ndomains

    def output_num(self):
        return self.ndomains

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

    def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
        return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

    def grl_hook(self, coeff):

        def fun1(grad):
            return -coeff * grad.clone()
        return fun1

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = self.calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        if self.training:
            x.register_hook(self.grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y


resnet_dict = {'ResNet18': models.resnet18, 'ResNet34': models.resnet34, 'ResNet50': models.resnet50, 'ResNet101': models.resnet101, 'ResNet152': models.resnet152}


class ResNetFc(nn.Module):

    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2}, {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2}, {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
            else:
                parameter_list = [{'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2}, {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
        else:
            parameter_list = [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        return parameter_list


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClassifierGNN,
     lambda: ([], {'in_features': 4, 'edge_features': 4, 'nclasses': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (EdgeNet,
     lambda: ([], {'in_features': 4, 'num_features': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (NodeNet,
     lambda: ([], {'in_features': 4, 'num_features': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (RandomLayer,
     lambda: ([], {'input_dim_list': [4, 4], 'output_dim': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_Evgeneus_Graph_Domain_Adaptaion(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

