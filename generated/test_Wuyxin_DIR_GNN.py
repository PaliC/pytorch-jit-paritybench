import sys
_module = sys.modules[__name__]
del sys
datasets = _module
graphsst2_dataset = _module
mnistsp_dataset = _module
spmotif_dataset = _module
gnn = _module
graphsst2_gnn = _module
mnistsp_gnn = _module
molhiv_gnn = _module
overloader = _module
spmotif_gnn = _module
BA3_loc = _module
featgen = _module
synthetic_structsim = _module
mnistsp_dir = _module
molhiv_dir = _module
spmotif_dir = _module
sst2_dir = _module
utils = _module
assigner = _module
dro_loss = _module
get_subgraph = _module
helper = _module
logger = _module
mask = _module
saver = _module

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


import numpy as np


from torch.utils.data import random_split


from torch.utils.data import Subset


import torch.utils


import torch.utils.data


import torch.nn.functional as F


from scipy.spatial.distance import cdist


import time


from torch.nn import Linear


from torch.nn import ReLU


from torch.nn import CrossEntropyLoss


from torch.optim.lr_scheduler import ReduceLROnPlateau


from collections import OrderedDict


from torch.nn import ModuleList


import torch.nn as nn


from torch.nn import Sequential as Seq


from torch.nn import Tanh


from torch.nn import Linear as Lin


from torch.nn import Softmax


import copy


from torch.autograd import grad


import math


from torch import Tensor


import re


class GraphSST2Net(torch.nn.Module):

    def __init__(self, in_channels, hid_channels=128, num_classes=2, num_layers=1):
        super(GraphSST2Net, self).__init__()
        self.convs = ModuleList([ARMAConv(in_channels, hid_channels), ARMAConv(hid_channels, hid_channels, num_layers=num_layers)])
        self.causal_mlp = torch.nn.Sequential(Linear(hid_channels, 2 * hid_channels), ReLU(), Linear(2 * hid_channels, num_classes))
        self.conf_mlp = torch.nn.Sequential(Linear(hid_channels, 2 * hid_channels), ReLU(), Linear(2 * hid_channels, num_classes))

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_causal_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        x = F.relu(self.convs[0](x, edge_index, edge_weight))
        node_x = self.convs[1](x, edge_index, edge_weight)
        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_causal_pred(self, graph_x):
        return self.causal_mlp(graph_x)

    def get_conf_pred(self, graph_x):
        return self.conf_mlp(graph_x)

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_mlp(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


class MNISTSPNet(torch.nn.Module):

    def __init__(self, in_channels, hid_channels=32, num_classes=10, conv_unit=2):
        super(MNISTSPNet, self).__init__()
        self.node_emb = Lin(in_channels, hid_channels)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()
        for i in range(conv_unit):
            conv = GraphConv(in_channels=hid_channels, out_channels=hid_channels)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hid_channels))
            self.relus.append(ReLU())
        self.causal_mlp = nn.Sequential(nn.Linear(hid_channels, 2 * hid_channels), nn.ReLU(), nn.Linear(2 * hid_channels, num_classes))
        self.conf_mlp = torch.nn.Sequential(nn.Linear(hid_channels, 2 * hid_channels), ReLU(), nn.Linear(2 * hid_channels, num_classes))

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_max_pool(node_x, batch)
        return self.get_causal_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        x = self.node_emb(x)
        for conv, batch_norm, ReLU in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = ReLU(batch_norm(x))
        node_x = x
        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_max_pool(node_x, batch)
        return graph_x

    def get_causal_pred(self, causal_graph_x):
        pred = self.causal_mlp(causal_graph_x)
        return pred

    def get_conf_pred(self, conf_graph_x):
        pred = self.conf_mlp(conf_graph_x)
        return pred

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_mlp(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred


class GINVirtual_node(torch.nn.Module):
    """
    Helper function of Graph Isomorphism Network augmented with virtual node for multi-task binary graph classification
    This will generate node embeddings
    Input:
        - batched Pytorch Geometric graph object
    Output:
        - node_embedding (Tensor): float torch tensor of shape (num_nodes, emb_dim)
    """

    def __init__(self, num_layers, emb_dim, dropout=0.5, encode_node=True):
        """
        Args:
            - num_tasks (int): number of binary label tasks. default to 128 (number of tasks of ogbg-molpcba)
            - num_layers (int): number of message passing layers of GNN
            - emb_dim (int): dimensionality of hidden channels
            - dropout (float): dropout ratio applied to hidden channels
        """
        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_node = encode_node
        if self.num_layers < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')
        self.atom_encoder = AtomEncoder(emb_dim)
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def forward(self, x, edge_index, edge_attr, batch):
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype))
        if self.encode_node:
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]
        for layer in range(self.num_layers):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)
            if layer < self.num_layers - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.dropout, training=self.training)
        node_embedding = h_list[-1]
        return node_embedding


class MolHivNet(torch.nn.Module):
    """
    Graph Isomorphism Network augmented with virtual node for multi-task binary graph classification
    Input:
        - batched Pytorch Geometric graph object
    Output:
        - prediction (Tensor): float torch tensor of shape (num_graphs, num_tasks)
    """

    def __init__(self, num_tasks=1, num_layers=3, emb_dim=300, dropout=0.5, encode_node=False):
        """
        Args:
            - num_tasks (int): number of binary label tasks. default to 128 (number of tasks of ogbg-molpcba)
            - num_layers (int): number of message passing layers of GNN
            - emb_dim (int): dimensionality of hidden channels
            - dropout (float): dropout ratio applied to hidden channels
        """
        super(MolHivNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.d_out = self.num_tasks
        if self.num_layers < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')
        self.gnn_node = GINVirtual_node(num_layers, emb_dim, dropout=dropout, encode_node=encode_node)
        self.pool = global_mean_pool
        self.causal_lin = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.conf_lin = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.cq = torch.nn.Linear(self.num_tasks, self.num_tasks)

    def forward(self, x, edge_index, edge_attr, batch):
        h_graph = self.get_graph_rep(x, edge_index, edge_attr, batch)
        return self.get_causal_pred(h_graph)

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        h_node = self.gnn_node(x, edge_index, edge_attr, batch)
        h_graph = self.pool(h_node, batch)
        return h_graph

    def get_causal_pred(self, h_graph):
        return self.causal_lin(h_graph)

    def get_conf_pred(self, conf_graph_x):
        return self.conf_lin(conf_graph_x)

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_lin(causal_graph_x)
        conf_pred = self.conf_lin(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred


def overload(func):

    @wraps(func)
    def wrapper(*args, **kargs):
        if len(args) + len(kargs) == 2:
            if len(args) == 2:
                g = args[1]
            else:
                g = kargs['graph']
            return func(args[0], g.x, g.edge_index, g.edge_attr, g.batch)
        elif len(args) + len(kargs) == 5:
            if len(args) == 5:
                return func(*args)
            else:
                return func(args[0], **kargs)
        elif len(args) + len(kargs) == 6:
            if len(args) == 6:
                return func(*args[:-1])
            else:
                return func(args[0], kargs['x'], kargs['edge_index'], kargs['edge_attr'], kargs['batch'])
        else:
            raise TypeError
    return wrapper


class SPMotifNet(torch.nn.Module):

    def __init__(self, in_channels, hid_channels=64, num_classes=3, num_unit=2):
        super().__init__()
        self.num_unit = num_unit
        self.node_emb = Linear(in_channels, hid_channels)
        self.convs = ModuleList()
        self.relus = ModuleList()
        for i in range(num_unit):
            conv = LEConv(in_channels=hid_channels, out_channels=hid_channels)
            self.convs.append(conv)
            self.relus.append(ReLU())
        self.causal_mlp = torch.nn.Sequential(Linear(hid_channels, 2 * hid_channels), ReLU(), Linear(2 * hid_channels, num_classes))
        self.conf_mlp = torch.nn.Sequential(Linear(hid_channels, 2 * hid_channels), ReLU(), Linear(2 * hid_channels, 3))
        self.cq = Linear(3, 3)
        self.conf_fw = torch.nn.Sequential(self.conf_mlp, self.cq)

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_causal_pred(graph_x)

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        for conv, ReLU in zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = ReLU(x)
        node_x = x
        return node_x

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_causal_pred(self, causal_graph_x):
        pred = self.causal_mlp(causal_graph_x)
        return pred

    def get_conf_pred(self, conf_graph_x):
        pred = self.conf_fw(conf_graph_x)
        return pred

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_mlp(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])
    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges


def split_graph(data, edge_score, ratio):
    causal_edge_index = torch.LongTensor([[], []])
    causal_edge_weight = torch.tensor([])
    causal_edge_attr = torch.tensor([])
    conf_edge_index = torch.LongTensor([[], []])
    conf_edge_weight = torch.tensor([])
    conf_edge_attr = torch.tensor([])
    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve = int(ratio * N)
        edge_attr = data.edge_attr[C:C + N]
        single_mask = edge_score[C:C + N]
        single_mask_detach = edge_score[C:C + N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
        causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
        conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)
        causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
        conf_edge_weight = torch.cat([conf_edge_weight, -1 * single_mask[idx_drop]])
        causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
        conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])
    return (causal_edge_index, causal_edge_attr, causal_edge_weight), (conf_edge_index, conf_edge_attr, conf_edge_weight)


class CausalAttNet(nn.Module):

    def __init__(self, causal_ratio):
        super(CausalAttNet, self).__init__()
        self.conv1 = ARMAConv(in_channels=768, out_channels=args.channels)
        self.conv2 = ARMAConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(nn.Linear(args.channels * 2, args.channels * 4), nn.ReLU(), nn.Linear(args.channels * 4, 1))
        self.ratio = causal_ratio

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))
        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)
        (causal_edge_index, causal_edge_attr, causal_edge_weight), (conf_edge_index, conf_edge_attr, conf_edge_weight) = split_graph(data, edge_score, self.ratio)
        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)
        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), edge_score

