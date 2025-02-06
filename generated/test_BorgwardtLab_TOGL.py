import sys
_module = sys.modules[__name__]
del sys
datagen = _module
deps = _module
test_Benchmarking_gnns_metrics = _module
topognn = _module
adjacency_matrix_to_edge_list = _module
analyse_graphs_ph = _module
analyse_graphs_wl = _module
analyse_persistent_homology_statistics = _module
cli_utils = _module
convert_graphs = _module
coord_transforms = _module
data_utils = _module
format_output = _module
gcn = _module
graph6_to_edge_lists = _module
layers = _module
metrics = _module
models = _module
plot_graphs = _module
rerun_run = _module
simple = _module
synthetic_gcn = _module
synthetic_topognn = _module
tasks = _module
train_model = _module
tu_datasets = _module
weisfeiler_lehman = _module

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


from sklearn.metrics import confusion_matrix


from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import cross_val_score


from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import StratifiedKFold


from sklearn.svm import SVC


from sklearn.metrics.pairwise import euclidean_distances


import matplotlib.pyplot as plt


import torch.nn as nn


import math


import torch.nn.functional as F


import itertools


from torch.utils.data import random_split


from torch.utils.data import Subset


from sklearn.model_selection import train_test_split


from typing import Any


from typing import Callable


from typing import Optional


from sklearn.metrics import roc_auc_score


class Triangle_transform(nn.Module):

    def __init__(self, output_dim):
        """
        output dim is the number of t parameters in the triangle point transformation
        """
        super().__init__()
        self.output_dim = output_dim
        self.t_param = torch.nn.Parameter(torch.randn(output_dim) * 0.1, requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """
        return torch.nn.functional.relu(x[:, 1][:, None] - torch.abs(self.t_param - x[:, 0][:, None]))


class Gaussian_transform(nn.Module):

    def __init__(self, output_dim):
        """
        output dim is the number of t parameters in the Gaussian point transformation
        """
        super().__init__()
        self.output_dim = output_dim
        self.t_param = torch.nn.Parameter(torch.randn(output_dim) * 0.1, requires_grad=True)
        self.sigma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """
        return torch.exp(-(x[:, :, None] - self.t_param).pow(2).sum(axis=1) / (2 * self.sigma.pow(2)))


class Line_transform(nn.Module):

    def __init__(self, output_dim):
        """
        output dim is the number of lines in the Line point transformation
        """
        super().__init__()
        self.output_dim = output_dim
        self.lin_mod = torch.nn.Linear(2, output_dim)

    def forward(self, x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """
        return self.lin_mod(x)


class RationalHat_transform(nn.Module):
    """
    Coordinate function as defined in 

    /Hofer, C., Kwitt, R., and Niethammer, M.
    Learning representations of persistence barcodes.
    JMLR, 20(126):1–45, 2019b./

    """

    def __init__(self, output_dim, input_dim=1):
        """
        output dim is the number of lines in the Line point transformation
        """
        super().__init__()
        self.output_dim = output_dim
        self.c_param = torch.nn.Parameter(torch.randn(input_dim, output_dim) * 0.1, requires_grad=True)
        self.r_param = torch.nn.Parameter(torch.randn(1, output_dim) * 0.1, requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,input_dim]
        output is of shape [N,output_dim]
        """
        first_element = 1 + torch.norm(x[:, :, None] - self.c_param, p=1, dim=1)
        second_element = 1 + torch.abs(torch.abs(self.r_param) - torch.norm(x[:, :, None] - self.c_param, p=1, dim=1))
        return 1 / first_element - 1 / second_element


class MAB(nn.Module):

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None):
        """
        mask should be of shape [batch, length]
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        if mask is not None:
            mask_repeat = mask[:, None, :].repeat(self.num_heads, Q.shape[1], 1)
            before_softmax = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
            before_softmax[~mask_repeat] = -10000000000.0
        else:
            before_softmax = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        A = torch.softmax(before_softmax, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
        return self.mab1(X, H)


def batch_to_tensor(batch, external_tensor, attribute='x'):
    """
    Takes a pytorch geometric batch and returns the data as a regular tensor padded with 0 and the associated mask
    stacked_tensor [Num graphs, Max num nodes, D]
    mask [Num_graphs, Max num nodes]
    """
    batch_list = []
    idx = batch.__slices__[attribute]
    for i in range(1, 1 + len(batch.y)):
        batch_list.append(external_tensor[idx[i - 1]:idx[i]])
    stacked_tensor = torch.nn.utils.rnn.pad_sequence(batch_list, batch_first=True)
    mask = torch.zeros(stacked_tensor.shape[:2])
    for i in range(1, 1 + len(batch.y)):
        mask[i - 1, :idx[i] - idx[i - 1]] = 1
    mask_zeros = (stacked_tensor != 0).any(2)
    return stacked_tensor, mask, mask_zeros


class Set2SetMod(torch.nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, num_inds):
        super().__init__()
        self.set_transform = ISAB(dim_in=dim_in, dim_out=dim_out, num_heads=num_heads, num_inds=num_inds)

    def forward(self, x, batch, dim1_flag=False):
        if dim1_flag:
            stacked_tensor, mask, mask_zeros = batch_to_tensor(batch, x, attribute='edge_index')
            out_ = self.set_transform(stacked_tensor, mask)
            out_[mask_zeros] = 0
            out = out_[mask]
        else:
            stacked_tensor, mask, mask_zeros = batch_to_tensor(batch, x)
            out_ = self.set_transform(stacked_tensor, mask)
            out = out_[mask]
        return out


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, activation, dropout, batch_norm, residual=True):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        self.conv = GCNConv(in_features, out_features, add_self_loops=False)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class GINLayer(nn.Module):

    def __init__(self, in_features, out_features, activation, dropout, batch_norm, mlp_hidden_dim=None, residual=True, train_eps=False, **kwargs):
        super().__init__()
        if mlp_hidden_dim is None:
            mlp_hidden_dim = in_features
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        gin_net = nn.Sequential(nn.Linear(in_features, mlp_hidden_dim), nn.ReLU(), nn.Linear(mlp_hidden_dim, out_features))
        self.conv = GINConv(gin_net, train_eps=train_eps)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, activation, dropout, batch_norm, num_heads, residual=True, train_eps=False, **kwargs):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_features * num_heads) if batch_norm else nn.Identity()
        self.conv = GATConv(in_features, out_features, heads=num_heads, dropout=dropout)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class GatedGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, activation, dropout, batch_norm, residual=True, train_eps=False, **kwargs):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        self.conv = ResGatedGraphConv(in_features, out_features)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim, aggregation_fn):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ['mean', 'max', 'sum']
        self.aggregation_fn = aggregation_fn

    def forward(self, x, batch):
        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm[batch, :]
        return x


class DeepSetLayerDim1(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim, aggregation_fn):
        super().__init__()
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ['mean', 'max', 'sum']
        self.aggregation_fn = aggregation_fn

    def forward(self, x, edge_slices, mask=None):
        """
        Mask is True where the persistence (x) is observed.
        """
        edge_diff_slices = edge_slices[1:] - edge_slices[:-1]
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(torch.arange(n_batch, device=x.device), edge_diff_slices)
        if mask is not None:
            batch_e = batch_e[mask]
        xm = scatter(x, batch_e, dim=0, reduce=self.aggregation_fn, dim_size=n_batch)
        xm = self.Lambda(xm)
        return xm


def fake_persistence_computation(filtered_v_, edge_index, vertex_slices, edge_slices, batch):
    device = filtered_v_.device
    num_filtrations = filtered_v_.shape[1]
    filtered_e_, _ = torch.max(torch.stack((filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)
    persistence0_new = filtered_v_.unsqueeze(-1).expand(-1, -1, 2)
    edge_slices = edge_slices
    bs = edge_slices.shape[0] - 1
    unpaired_values = torch.zeros((bs, num_filtrations), device=device)
    persistence1_new = torch.zeros(edge_index.shape[1], filtered_v_.shape[1], 2, device=device)
    n_edges = edge_slices[1:] - edge_slices[:-1]
    random_edges = (edge_slices[0:-1].unsqueeze(-1) + torch.floor(torch.rand(size=(bs, num_filtrations), device=device) * n_edges.float().unsqueeze(-1))).long()
    persistence1_new[random_edges, torch.arange(num_filtrations).unsqueeze(0), :] = torch.stack([unpaired_values, filtered_e_[random_edges, torch.arange(num_filtrations).unsqueeze(0)]], -1)
    return persistence0_new.permute(1, 0, 2), persistence1_new.permute(1, 0, 2), None


def remove_duplicate_edges(batch):
    with torch.no_grad():
        batch = batch.clone()
        device = batch.x.device
        edge_slices = torch.tensor(batch.__slices__['edge_index'], device=device)
        edge_diff_slices = edge_slices[1:] - edge_slices[:-1]
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(torch.arange(n_batch, device=device), edge_diff_slices)
        correct_idx = batch.edge_index[0] <= batch.edge_index[1]
        n_edges = scatter(correct_idx.long(), batch_e, reduce='sum')
        batch.edge_index = batch.edge_index[:, correct_idx]
        new_slices = torch.cumsum(torch.cat((torch.zeros(1, device=device, dtype=torch.long), n_edges)), 0).tolist()
        batch.__slices__['edge_index'] = new_slices
        return batch


class SimpleSetTopoLayer(nn.Module):

    def __init__(self, n_features, n_filtrations, mlp_hidden_dim, aggregation_fn, dim0_out_dim, dim1_out_dim, dim1, residual_and_bn, fake, deepset_type='full', swap_bn_order=False, dist_dim1=False):
        super().__init__()
        self.filtrations = nn.Sequential(nn.Linear(n_features, mlp_hidden_dim), nn.ReLU(), nn.Linear(mlp_hidden_dim, n_filtrations))
        assert deepset_type in ['linear', 'shallow', 'full']
        self.num_filtrations = n_filtrations
        self.residual_and_bn = residual_and_bn
        self.swap_bn_order = swap_bn_order
        self.dist_dim1 = dist_dim1
        self.dim1_flag = dim1
        if self.dim1_flag:
            self.set_fn1 = nn.ModuleList([nn.Linear(n_filtrations * 2, dim1_out_dim), nn.ReLU(), DeepSetLayerDim1(in_dim=dim1_out_dim, out_dim=n_features if residual_and_bn and dist_dim1 else dim1_out_dim, aggregation_fn=aggregation_fn)])
        if deepset_type == 'linear':
            self.set_fn0 = nn.ModuleList([nn.Linear(n_filtrations * 2, n_features if residual_and_bn else dim0_out_dim, aggregation_fn)])
        elif deepset_type == 'shallow':
            self.set_fn0 = nn.ModuleList([nn.Linear(n_filtrations * 2, dim0_out_dim), nn.ReLU(), DeepSetLayer(dim0_out_dim, n_features if residual_and_bn else dim0_out_dim, aggregation_fn)])
        else:
            self.set_fn0 = nn.ModuleList([nn.Linear(n_filtrations * 2, dim0_out_dim), nn.ReLU(), DeepSetLayer(dim0_out_dim, dim0_out_dim, aggregation_fn), nn.ReLU(), DeepSetLayer(dim0_out_dim, n_features if residual_and_bn else dim0_out_dim, aggregation_fn)])
        if residual_and_bn:
            self.bn = nn.BatchNorm1d(n_features)
        elif dist_dim1:
            self.out = nn.Sequential(nn.Linear(dim0_out_dim + dim1_out_dim + n_features, n_features), nn.ReLU())
        else:
            self.out = nn.Sequential(nn.Linear(dim0_out_dim + n_features, n_features), nn.ReLU())
        self.fake = fake

    def compute_persistence(self, x, edge_index, vertex_slices, edge_slices, batch, return_filtration=False):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        filtered_v = self.filtrations(x)
        if self.fake:
            return fake_persistence_computation(filtered_v, edge_index, vertex_slices, edge_slices, batch)
        filtered_e, _ = torch.max(torch.stack((filtered_v[edge_index[0]], filtered_v[edge_index[1]])), axis=0)
        filtered_v = filtered_v.transpose(1, 0).cpu().contiguous()
        filtered_e = filtered_e.transpose(1, 0).cpu().contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()
        persistence0_new, persistence1_new = compute_persistence_homology_batched_mt(filtered_v, filtered_e, edge_index, vertex_slices, edge_slices)
        persistence0 = persistence0_new
        persistence1 = persistence1_new
        if return_filtration:
            return persistence0, persistence1, filtered_v
        else:
            return persistence0, persistence1, None

    def forward(self, x, data, return_filtration):
        data = remove_duplicate_edges(data)
        edge_index = data.edge_index
        vertex_slices = torch.Tensor(data.__slices__['x']).cpu().long()
        edge_slices = torch.Tensor(data.__slices__['edge_index']).cpu().long()
        batch = data.batch
        pers0, pers1, filtration = self.compute_persistence(x, edge_index, vertex_slices, edge_slices, batch, return_filtration)
        x0 = pers0.permute(1, 0, 2).reshape(pers0.shape[1], -1)
        for layer in self.set_fn0:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)
        if self.dim1_flag:
            pers1_reshaped = pers1.permute(1, 0, 2).reshape(pers1.shape[1], -1)
            pers1_mask = ~(pers1_reshaped == 0).all(-1)
            x1 = pers1_reshaped[pers1_mask]
            for layer in self.set_fn1:
                if isinstance(layer, DeepSetLayerDim1):
                    x1 = layer(x1, edge_slices, mask=pers1_mask)
                else:
                    x1 = layer(x1)
        else:
            x1 = None
        if self.residual_and_bn:
            if self.dist_dim1 and self.dim1_flag:
                x0 = x0 + x1[batch]
                x1 = None
            if self.swap_bn_order:
                x = x + F.relu(self.bn(x0))
            else:
                x = x + self.bn(F.relu(x0))
        else:
            if self.dist_dim1 and self.dim1_flag:
                x0 = torch.cat([x0, x1[batch]], dim=-1)
                x1 = None
            x = self.out(torch.cat([x, x0], dim=-1))
        return x, x1, filtration


class TopologyLayer(torch.nn.Module):
    """Topological Aggregation Layer."""

    def __init__(self, features_in, features_out, num_filtrations, num_coord_funs, filtration_hidden, num_coord_funs1=None, dim1=False, residual_and_bn=False, share_filtration_parameters=False, fake=False, tanh_filtrations=False, swap_bn_order=False, dist_dim1=False):
        """
        num_coord_funs is a dictionary with the numbers of coordinate functions of each type.
        dim1 is a boolean. True if we have to return dim1 persistence.
        """
        super().__init__()
        self.dim1 = dim1
        self.features_in = features_in
        self.features_out = features_out
        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs
        self.filtration_hidden = filtration_hidden
        self.residual_and_bn = residual_and_bn
        self.share_filtration_parameters = share_filtration_parameters
        self.fake = fake
        self.swap_bn_order = swap_bn_order
        self.dist_dim1 = dist_dim1
        self.total_num_coord_funs = np.array(list(num_coord_funs.values())).sum()
        self.coord_fun_modules = torch.nn.ModuleList([getattr(coord_transforms, key)(output_dim=num_coord_funs[key]) for key in num_coord_funs])
        if self.dim1:
            assert num_coord_funs1 is not None
            self.coord_fun_modules1 = torch.nn.ModuleList([getattr(coord_transforms, key)(output_dim=num_coord_funs1[key]) for key in num_coord_funs1])
        final_filtration_activation = nn.Tanh() if tanh_filtrations else nn.Identity()
        if self.share_filtration_parameters:
            self.filtration_modules = torch.nn.Sequential(torch.nn.Linear(self.features_in, self.filtration_hidden), torch.nn.ReLU(), torch.nn.Linear(self.filtration_hidden, num_filtrations), final_filtration_activation)
        else:
            self.filtration_modules = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.features_in, self.filtration_hidden), torch.nn.ReLU(), torch.nn.Linear(self.filtration_hidden, 1), final_filtration_activation) for _ in range(num_filtrations)])
        if self.residual_and_bn:
            in_out_dim = self.num_filtrations * self.total_num_coord_funs
            features_out = features_in
            self.bn = nn.BatchNorm1d(features_out)
            if self.dist_dim1 and self.dim1:
                self.out1 = torch.nn.Linear(self.num_filtrations * self.total_num_coord_funs, features_out)
        elif self.dist_dim1:
            in_out_dim = self.features_in + 2 * self.num_filtrations * self.total_num_coord_funs
        else:
            in_out_dim = self.features_in + self.num_filtrations * self.total_num_coord_funs
        self.out = torch.nn.Linear(in_out_dim, features_out)

    def compute_persistence(self, x, batch, return_filtration=False):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        edge_index = batch.edge_index
        if self.share_filtration_parameters:
            filtered_v_ = self.filtration_modules(x)
        else:
            filtered_v_ = torch.cat([filtration_mod.forward(x) for filtration_mod in self.filtration_modules], 1)
        filtered_e_, _ = torch.max(torch.stack((filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)
        vertex_slices = torch.Tensor(batch.__slices__['x']).long()
        edge_slices = torch.Tensor(batch.__slices__['edge_index']).long()
        if self.fake:
            return fake_persistence_computation(filtered_v_, edge_index, vertex_slices, edge_slices, batch.batch)
        vertex_slices = vertex_slices.cpu()
        edge_slices = edge_slices.cpu()
        filtered_v_ = filtered_v_.cpu().transpose(1, 0).contiguous()
        filtered_e_ = filtered_e_.cpu().transpose(1, 0).contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()
        persistence0_new, persistence1_new = compute_persistence_homology_batched_mt(filtered_v_, filtered_e_, edge_index, vertex_slices, edge_slices)
        persistence0_new = persistence0_new
        persistence1_new = persistence1_new
        if return_filtration:
            return persistence0_new, persistence1_new, filtered_v_
        else:
            return persistence0_new, persistence1_new, None

    def compute_coord_fun(self, persistence, batch, dim1=False):
        """
        Input : persistence [N_points,2]
        Output : coord_fun mean-aggregated [self.num_coord_fun]
        """
        if dim1:
            coord_activation = torch.cat([mod.forward(persistence) for mod in self.coord_fun_modules1], 1)
        else:
            coord_activation = torch.cat([mod.forward(persistence) for mod in self.coord_fun_modules], 1)
        return coord_activation

    def compute_coord_activations(self, persistences, batch, dim1=False):
        """
        Return the coordinate functions activations pooled by graph.
        Output dims : list of length number of filtrations with elements : [N_graphs in batch, number fo coordinate functions]
        """
        coord_activations = [self.compute_coord_fun(persistence, batch=batch, dim1=dim1) for persistence in persistences]
        return torch.cat(coord_activations, 1)

    def collapse_dim1(self, activations, mask, slices):
        """
        Takes a flattened tensor of activations along with a mask and collapses it (sum) to have a graph-wise features

        Inputs : 
        * activations [N_edges,d]
        * mask [N_edge]
        * slices [N_graphs]
        Output:
        * collapsed activations [N_graphs,d]
        """
        collapsed_activations = []
        for el in range(len(slices) - 1):
            activations_el_ = activations[slices[el]:slices[el + 1]]
            mask_el = mask[slices[el]:slices[el + 1]]
            activations_el = activations_el_[mask_el].sum(axis=0)
            collapsed_activations.append(activations_el)
        return torch.stack(collapsed_activations)

    def forward(self, x, batch, return_filtration=False):
        batch = remove_duplicate_edges(batch)
        persistences0, persistences1, filtration = self.compute_persistence(x, batch, return_filtration)
        coord_activations = self.compute_coord_activations(persistences0, batch)
        if self.dim1:
            persistence1_mask = (persistences1 != 0).any(2).any(0)
            coord_activations1 = self.compute_coord_activations(persistences1, batch, dim1=True)
            graph_activations1 = self.collapse_dim1(coord_activations1, persistence1_mask, batch.__slices__['edge_index'])
        else:
            graph_activations1 = None
        if self.residual_and_bn:
            out_activations = self.out(coord_activations)
            if self.dim1 and self.dist_dim1:
                out_activations += self.out1(graph_activations1)[batch]
                graph_activations1 = None
            if self.swap_bn_order:
                out_activations = self.bn(out_activations)
                out_activations = x + F.relu(out_activations)
            else:
                out_activations = self.bn(out_activations)
                out_activations = x + out_activations
        else:
            concat_activations = torch.cat((x, coord_activations), 1)
            out_activations = self.out(concat_activations)
            out_activations = F.relu(out_activations)
        return out_activations, graph_activations1, filtration


class PointWiseMLP(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, x, **kwargs):
        return self.mlp(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Gaussian_transform,
     lambda: ([], {'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MAB,
     lambda: ([], {'dim_Q': 4, 'dim_K': 4, 'dim_V': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PointWiseMLP,
     lambda: ([], {'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RationalHat_transform,
     lambda: ([], {'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Triangle_transform,
     lambda: ([], {'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_BorgwardtLab_TOGL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

