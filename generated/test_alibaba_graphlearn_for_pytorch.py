import sys
_module = sys.modules[__name__]
del sys
bench_dist_neighbor_loader = _module
bench_feature = _module
bench_sampler = _module
run_dist_bench = _module
conf = _module
dist_sage_unsup = _module
preprocess_template = _module
dist_train_sage_supervised = _module
partition_ogbn_dataset = _module
run_dist_train_sage_sup = _module
sage_supervised_client = _module
sage_supervised_server = _module
feature_mp = _module
arxiv = _module
utils = _module
graph_sage_unsup_ppi = _module
bipartite_sage_unsup = _module
hierarchical_sage = _module
train_hgt_mag = _module
train_hgt_mag_mp = _module
igbh = _module
build_partition_feature = _module
compress_graph = _module
dataset = _module
dist_train_rgnn = _module
download = _module
mlperf_logging_utils = _module
partition = _module
rgnn = _module
split_seeds = _module
train_rgnn_multi_gpu = _module
utilities = _module
train_sage_ogbn_papers100m = _module
data_preprocess = _module
dist_train_products_sage = _module
train_products_sage = _module
seal_link_pred = _module
train_sage_ogbn_products = _module
train_sage_prod_with_trim = _module
python = _module
channel = _module
base = _module
mp_channel = _module
remote_channel = _module
shm_channel = _module
data = _module
dataset = _module
feature = _module
graph = _module
reorder = _module
table_dataset = _module
unified_tensor = _module
vineyard_utils = _module
distributed = _module
dist_client = _module
dist_context = _module
dist_dataset = _module
dist_feature = _module
dist_graph = _module
dist_link_neighbor_loader = _module
dist_loader = _module
dist_neighbor_loader = _module
dist_neighbor_sampler = _module
dist_options = _module
dist_random_partitioner = _module
dist_sampling_producer = _module
dist_server = _module
dist_subgraph_loader = _module
dist_table_dataset = _module
event_loop = _module
rpc = _module
loader = _module
link_loader = _module
link_neighbor_loader = _module
neighbor_loader = _module
node_loader = _module
subgraph_loader = _module
transform = _module
base = _module
frequency_partitioner = _module
partition_book = _module
random_partitioner = _module
sampler = _module
base = _module
negative_sampler = _module
neighbor_sampler = _module
typing = _module
build_glt = _module
common = _module
device = _module
exit_status = _module
mixin = _module
singleton = _module
tensor = _module
topo = _module
units = _module
setup = _module
dist_test_utils = _module
test_dist_feature = _module
test_dist_link_loader = _module
test_dist_neighbor_loader = _module
test_dist_random_partitioner = _module
test_dist_subgraph_loader = _module
test_feature = _module
test_graph = _module
test_hetero_neighbor_sampler = _module
test_link_loader = _module
test_neighbor_sampler = _module
test_partition = _module
test_pyg_remote_backend = _module
test_sample_prob = _module
test_shm_channel = _module
test_subgraph = _module
test_unified_tensor = _module
test_vineyard = _module

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


import torch.distributed as dist


import torch.distributed


import torch.nn.functional as F


from sklearn.metrics import roc_auc_score


from sklearn.metrics import recall_score


from torch.nn.parallel import DistributedDataParallel


from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer as ZeRO


from torch.distributed.algorithms.join import Join


from typing import Tuple


from typing import Optional


from typing import List


import torch.multiprocessing as mp


from sklearn.linear_model import SGDClassifier


from sklearn.metrics import f1_score


from sklearn.multioutput import MultiOutputClassifier


from torch.nn import Embedding


from torch.nn import Linear


import numpy as np


from typing import Literal


import sklearn.metrics


import warnings


from numpy import genfromtxt


import math


from itertools import chain


from scipy.sparse.csgraph import shortest_path


from torch.nn import BCEWithLogitsLoss


from torch.nn import Conv1d


from torch.nn import MaxPool1d


from torch.nn import ModuleList


from abc import ABC


from abc import abstractmethod


from typing import Dict


import logging


import queue


from typing import Union


from collections.abc import Sequence


from typing import Callable


from typing import Any


from enum import Enum


from torch._C import _set_worker_signal_handlers


from torch.utils.data.dataloader import DataLoader


import collections


import functools


from typing import Set


from torch.distributed import rpc


from typing import NamedTuple


from torch.utils.cpp_extension import CppExtension


import numpy


import random


from torch.utils.cpp_extension import BuildExtension


import re


from collections import defaultdict


class ItemGNNEncoder(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        item_x = self.conv1(x_dict['item'], edge_index_dict['item', 'to', 'item']).relu()
        user_x = self.conv2((x_dict['item'], x_dict['user']), edge_index_dict['item', 'rev_to', 'user']).relu()
        user_x = self.conv3((item_x, user_x), edge_index_dict['item', 'rev_to', 'user']).relu()
        return self.lin(user_x)


class EdgeDecoder(torch.nn.Module):

    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


device = torch.device(0)


edge_dir = 'out'


class Model(torch.nn.Module):

    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels, device=device)
        self.item_emb = Embedding(num_items, hidden_channels, device=device)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        x_dict['item'] = self.item_emb(x_dict['item'])
        z_dict['item'] = self.item_encoder(x_dict['item'], edge_index_dict['item', 'to', 'item'])
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)
        if edge_dir == 'out':
            return self.decoder(z_dict['item'], z_dict['user'], edge_label_index)
        return self.decoder(z_dict['user'], z_dict['item'], edge_label_index)


class HierarchicalHeteroGraphSage(torch.nn.Module):

    def __init__(self, edge_types, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in edge_types}, aggr='sum')
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict, num_sampled_nodes_dict):
        for i, conv in enumerate(self.convs):
            x_dict, edge_index_dict, _ = trim_to_layer(layer=i, num_sampled_nodes_per_hop=num_sampled_nodes_dict, num_sampled_edges_per_hop=num_sampled_edges_dict, x=x_dict, edge_index=edge_index_dict)
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['paper'])


class HGT(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, node_types, edge_types):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, (node_types, edge_types), num_heads, group='sum')
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return self.lin(x_dict['paper'])


class RGNN(torch.nn.Module):
    """ [Relational GNN model](https://arxiv.org/abs/1703.06103).

  Args:
    etypes: edge types.
    in_dim: input size.
    h_dim: Dimension of hidden layer.
    out_dim: Output dimension.
    num_layers: Number of conv layers.
    dropout: Dropout probability for hidden layers.
    model: "rsage" or "rgat".
    heads: Number of multi-head-attentions for GAT.
    node_type: The predict node type for node classification.

  """

    def __init__(self, etypes, in_dim, h_dim, out_dim, num_layers=2, dropout=0.2, model='rgat', heads=4, node_type=None, with_trim=False):
        super().__init__()
        self.node_type = node_type
        if node_type is not None:
            self.lin = torch.nn.Linear(h_dim, out_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_dim if i == 0 else h_dim
            h_dim = out_dim if i == num_layers - 1 and node_type is None else h_dim
            if model == 'rsage':
                self.convs.append(HeteroConv({etype: SAGEConv(in_dim, h_dim, root_weight=False) for etype in etypes}))
            elif model == 'rgat':
                self.convs.append(HeteroConv({etype: GATConv(in_dim, h_dim // heads, heads=heads, add_self_loops=False) for etype in etypes}))
        self.dropout = torch.nn.Dropout(dropout)
        self.with_trim = with_trim

    def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict=None, num_sampled_nodes_dict=None):
        for i, conv in enumerate(self.convs):
            if self.with_trim:
                x_dict, edge_index_dict, _ = trim_to_layer(layer=i, num_sampled_nodes_per_hop=num_sampled_nodes_dict, num_sampled_edges_per_hop=num_sampled_edges_dict, x=x_dict, edge_index=edge_index_dict)
            for key in list(edge_index_dict.keys()):
                if key[0] not in x_dict or key[-1] not in x_dict:
                    del edge_index_dict[key]
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs) - 1:
                x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        if hasattr(self, 'lin'):
            return self.lin(x_dict[self.node_type])
        else:
            return x_dict


class CastMixin:
    """ This class is same as PyG's :class:`~torch_geometric.utils.CastMixin`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/mixin.py
  """

    @classmethod
    def cast(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None
            if isinstance(elem, CastMixin):
                return elem
            if isinstance(elem, (tuple, list)):
                return cls(*elem)
            if isinstance(elem, dict):
                return cls(**elem)
        return cls(*args, **kwargs)


class EdgeIndex(NamedTuple):
    """ PyG's :class:`~torch_geometric.loader.EdgeIndex` used in old data loader
  :class:`~torch_geometric.loader.NeighborSampler`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/loader/neighbor_sampler.py
  """
    edge_index: 'torch.Tensor'
    e_id: 'Optional[torch.Tensor]'
    size: 'Tuple[int, int]'

    def to(self, *args, **kwargs):
        edge_index = self.edge_index
        e_id = self.e_id if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Graph(object):
    """ A graph object used for graph operations such as sampling.

  There are three modes supported:
    1.'CPU': graph data are stored in the CPU memory and graph
      operations are also executed on CPU.
    2.'ZERO_COPY': graph data are stored in the pinned CPU memory and graph
      operations are executed on GPU.
    3.'CUDA': graph data are stored in the GPU memory and graph operations
      are executed on GPU.

  Args:
    topo (Topology): An instance of ``Topology`` with graph topology data.
    mode (str): The graph operation mode, must be 'CPU', 'ZERO_COPY' or 'CUDA'.
      (Default: 'ZERO_COPY').
    device (int, optional): The target cuda device rank to perform graph
      operations. Note that this parameter will be ignored if the graph mode
      set to 'CPU'. The value of ``torch.cuda.current_device()`` will be used
      if set to ``None``. (Default: ``None``).
  """

    def __init__(self, topo: 'Topology', mode='ZERO_COPY', device: 'Optional[int]'=None):
        self.topo = topo
        self.topo.share_memory_()
        self.mode = mode.upper()
        self.device = device
        if self.mode != 'CPU' and self.device is not None:
            self.device = int(self.device)
            assert self.device >= 0 and self.device < torch.cuda.device_count(), f"'{self.__class__.__name__}': invalid device rank {self.device}"
        self._graph = None

    def lazy_init(self):
        if self._graph is not None:
            return
        self._graph = pywrap.Graph()
        indptr = self.topo.indptr
        indices = self.topo.indices
        if self.topo.edge_ids is not None:
            edge_ids = self.topo.edge_ids
        else:
            edge_ids = torch.empty(0)
        if self.topo.edge_weights is not None:
            edge_weights = self.topo.edge_weights
        else:
            edge_weights = torch.empty(0)
        if self.mode == 'CPU':
            self._graph.init_cpu_from_csr(indptr, indices, edge_ids, edge_weights)
        else:
            if self.device is None:
                self.device = torch.cuda.current_device()
            if self.mode == 'CUDA':
                self._graph.init_cuda_from_csr(indptr, indices, self.device, pywrap.GraphMode.DMA, edge_ids)
            elif self.mode == 'ZERO_COPY':
                self._graph.init_cuda_from_csr(indptr, indices, self.device, pywrap.GraphMode.ZERO_COPY, edge_ids)
            else:
                raise ValueError(f"'{self.__class__.__name__}': invalid mode {self.mode}")

    def export_topology(self):
        return self.topo.indptr, self.topo.indices, self.topo.edge_ids

    def share_ipc(self):
        """ Create ipc handle for multiprocessing.

    Returns:
      A tuple of topo and graph mode.
    """
        return self.topo, self.mode

    @classmethod
    def from_ipc_handle(cls, ipc_handle):
        """ Create from ipc handle.
    """
        topo, mode = ipc_handle
        return cls(topo, mode, device=None)

    @property
    def row_count(self):
        self.lazy_init()
        return self._graph.get_row_count()

    @property
    def col_count(self):
        self.lazy_init()
        return self._graph.get_col_count()

    @property
    def edge_count(self):
        self.lazy_init()
        return self._graph.get_edge_count()

    @property
    def graph_handler(self):
        """ Get a pointer to the underlying graph object for graph operations
    such as sampling.
    """
        self.lazy_init()
        return self._graph


NodeType = str


class RandomNegativeSampler(object):
    """ Random negative Sampler.

  Args:
    graph: A ``graphlearn_torch.data.Graph`` object.
    mode: Execution mode of sampling, 'CUDA' means sampling on
      GPU, 'CPU' means sampling on CPU.
    edge_dir: The direction of edges to be sampled, determines 
      the order of rows and columns returned.
  """

    def __init__(self, graph, mode='CUDA', edge_dir='out'):
        self._mode = mode
        self.edge_dir = edge_dir
        if mode == 'CUDA':
            self._sampler = pywrap.CUDARandomNegativeSampler(graph.graph_handler)
        else:
            self._sampler = pywrap.CPURandomNegativeSampler(graph.graph_handler)

    def sample(self, req_num, trials_num=5, padding=False):
        """ Negative sampling.

    Args:
      req_num: The number of request(max) negative samples.
      trials_num: The number of trials for negative sampling.
      padding: Whether to patch the negative sampling results to req_num.
        If True, after trying trials_num times, if the number of true negative
        samples is still less than req_num, just random sample edges(non-strict
        negative) as negative samples.

    Returns:
      negative edge_index(non-strict when padding is True).
    """
        if self.edge_dir == 'out':
            rows, cols = self._sampler.sample(req_num, trials_num, padding)
        elif self.edge_dir == 'in':
            cols, rows = self._sampler.sample(req_num, trials_num, padding)
        return torch.stack([rows, cols], dim=0)


def count_dict(in_dict: 'Dict[Any, Any]', out_dict: 'Dict[Any, Any]', target_len):
    for k, v in in_dict.items():
        vals = out_dict.get(k, [])
        vals += [0] * (target_len - len(vals) - 1)
        vals.append(len(v))
        out_dict[k] = vals


def reverse_edge_type(etype: 'EdgeType'):
    src, edge, dst = etype
    if not src == dst:
        if edge.split('_', 1)[0] == 'rev':
            edge = edge.split('_', 1)[1]
        else:
            edge = 'rev_' + edge
    return dst, edge, src


def format_hetero_sampler_output(in_sample: 'Any', edge_dir=Literal['in', 'out']):
    for k in in_sample.node.keys():
        in_sample.node[k] = in_sample.node[k].unique()
    if in_sample.edge_types is not None:
        if edge_dir == 'out':
            in_sample.edge_types = [(reverse_edge_type(etype) if etype[0] != etype[-1] else etype) for etype in in_sample.edge_types]
    return in_sample


def id2idx(ids: 'Union[List[int], torch.Tensor]'):
    """ Get tensor of mapping from id to its original index.
  """
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.int64)
    ids = ids
    max_id = torch.max(ids).item()
    id2idx = torch.zeros(max_id + 1, dtype=torch.int64, device=ids.device)
    id2idx[ids] = torch.arange(ids.size(0), dtype=torch.int64, device=ids.device)
    return id2idx


def merge_dict(in_dict: 'Dict[Any, Any]', out_dict: 'Dict[Any, Any]'):
    for k, v in in_dict.items():
        vals = out_dict.get(k, [])
        vals.append(v)
        out_dict[k] = vals


def merge_hetero_sampler_output(in_sample: 'Any', out_sample: 'Any', device, edge_dir: "Literal['in', 'out']"='out'):

    def subid2gid(sample):
        for k, v in sample.row.items():
            sample.row[k] = sample.node[k[0]][v]
        for k, v in sample.col.items():
            sample.col[k] = sample.node[k[-1]][v]

    def merge_tensor_dict(in_dict, out_dict, unique=False):
        for k, v in in_dict.items():
            vals = out_dict.get(k, torch.tensor([], device=device))
            out_dict[k] = torch.cat((vals, v)).unique() if unique else torch.cat((vals, v))
    subid2gid(in_sample)
    subid2gid(out_sample)
    merge_tensor_dict(in_sample.node, out_sample.node, unique=True)
    merge_tensor_dict(in_sample.row, out_sample.row)
    merge_tensor_dict(in_sample.col, out_sample.col)
    for k, v in out_sample.row.items():
        out_sample.row[k] = id2idx(out_sample.node[k[0]])[v]
    for k, v in out_sample.col.items():
        out_sample.col[k] = id2idx(out_sample.node[k[-1]])[v]
    if in_sample.edge is not None and out_sample.edge is not None:
        merge_tensor_dict(in_sample.edge, out_sample.edge, unique=False)
    if out_sample.edge_types is not None and in_sample.edge_types is not None:
        out_sample.edge_types = list(set(out_sample.edge_types) | set(in_sample.edge_types))
        if edge_dir == 'out':
            out_sample.edge_types = [(reverse_edge_type(etype) if etype[0] != etype[-1] else etype) for etype in out_sample.edge_types]
    return out_sample


class SAGE(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in test_loader:
                edge_index, _, size = adj
                total_edges += edge_index.size(1)
                x = x_all[n_id]
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

