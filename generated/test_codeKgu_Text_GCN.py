import sys
_module = sys.modules[__name__]
del sys
build_graph = _module
config = _module
dataset = _module
eval = _module
load_data = _module
main = _module
model_factory = _module
model_text_gnn = _module
prep_data = _module
saver = _module
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


import torch


import numpy as np


import random


import scipy.sparse as sp


from collections import Counter


import torch.nn.functional as F


import torch.nn as nn


from torch.nn import Parameter


import time


parser = argparse.ArgumentParser()


FLAGS = parser.parse_args()


def create_act(act, num_parameters=None):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity':


        class Identity(nn.Module):

            def forward(self, x):
                return x
        return Identity()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


class MLP(nn.Module):
    """mlp can specify number of hidden layers and hidden layer channels"""

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2, hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError('number of hidden layers should be the same as the lengh of hidden_channels')
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1]) for i in range(len(self.layer_channels) - 1)])))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]


class NodeEmbedding(nn.Module):

    def __init__(self, type, in_dim, out_dim, act, bn, dropout):
        super(NodeEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        if type == 'gcn':
            self.conv = GCNConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        elif type == 'gat':
            self.conv = GATConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        else:
            raise ValueError('Unknown node embedding layer type {}'.format(type))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        if dropout:
            self.dropout = torch.nn.Dropout()

    def forward(self, ins, pyg_graph):
        if self.dropout:
            ins = self.dropout(ins)
        if self.type == 'gcn':
            if FLAGS.use_edge_weights:
                x = self.conv(ins, pyg_graph.edge_index, edge_weight=pyg_graph.edge_attr)
            else:
                x = self.conv(ins, pyg_graph.edge_index)
        else:
            x = self.conv(ins, pyg_graph.edge_index)
        x = self.act(x)
        return x


class TextGNN(nn.Module):

    def __init__(self, pred_type, node_embd_type, num_layers, layer_dim_list, act, bn, num_labels, class_weights, dropout):
        super(TextGNN, self).__init__()
        self.node_embd_type = node_embd_type
        self.layer_dim_list = layer_dim_list
        self.num_layers = num_layers
        self.dropout = dropout
        if pred_type == 'softmax':
            assert layer_dim_list[-1] == num_labels
        elif pred_type == 'mlp':
            dims = self._calc_mlp_dims(layer_dim_list[-1], num_labels)
            self.mlp = MLP(layer_dim_list[-1], num_labels, num_hidden_lyr=len(dims), hidden_channels=dims, bn=False)
        self.pred_type = pred_type
        assert len(layer_dim_list) == num_layers + 1
        self.act = act
        self.bn = bn
        self.layers = self._create_node_embd_layers()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pyg_graph, dataset):
        acts = [pyg_graph.x]
        for i, layer in enumerate(self.layers):
            ins = acts[-1]
            outs = layer(ins, pyg_graph)
            acts.append(outs)
        return self._loss(acts[-1], dataset)

    def _loss(self, ins, dataset):
        pred_inds = dataset.node_ids
        if self.pred_type == 'softmax':
            y_preds = ins[pred_inds]
        elif self.pred_type == 'mlp':
            y_preds = self.mlp(ins[pred_inds])
        else:
            raise NotImplementedError
        y_true = torch.tensor(dataset.label_inds[pred_inds], dtype=torch.long, device=FLAGS.device)
        loss = self.loss(y_preds, y_true)
        return loss, y_preds.cpu().detach().numpy()

    def _create_node_embd_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            act = self.act if i < self.num_layers - 1 else 'identity'
            layers.append(NodeEmbedding(type=self.node_embd_type, in_dim=self.layer_dim_list[i], out_dim=self.layer_dim_list[i + 1], act=act, bn=self.bn, dropout=self.dropout if i != 0 else False))
        return layers

    def _calc_mlp_dims(self, mlp_dim, output_dim=1):
        dim = mlp_dim
        dims = []
        while dim > output_dim:
            dim = dim // 2
            dims.append(dim)
        dims = dims[:-1]
        return dims

