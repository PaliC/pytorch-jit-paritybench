import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
AMSGrad = _module
metrics = _module
stats_extractor = _module
transforms = _module
utils = _module
attention = _module
astgcn = _module
attention_model = _module
mstgcn = _module
seq2seq = _module
stgcn = _module
recurrent = _module
agcrn = _module
dcrnn = _module
decoder = _module
encoder = _module
gcgru = _module
seq2seq = _module
seq2seq_model = _module
temporalgcn = _module
clcnn = _module
attention_model = _module
clcstn = _module
clccell = _module
clconv = _module
graphconv = _module
decoder = _module
encoder = _module
seq2seq_model = _module
seq2seq = _module
loss = _module
sphere = _module
generate_training_data = _module
supervisor = _module
train_clcrn = _module
train_clcstn = _module

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


import logging


import numpy as np


import scipy.sparse as sp


import matplotlib.pyplot as plt


import torch


from scipy.sparse import linalg


from torch.utils.data import Dataset


import math


from typing import Optional


from typing import List


from typing import Union


import torch.nn as nn


from torch.nn import Parameter


import torch.nn.functional as F


import re


from typing import Text


from numpy.core.overrides import ArgSpec


import pandas as pd


import time


class SpatialAttention(nn.Module):
    """An implementation of the Spatial Attention Module. For details see this paper: 
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow 
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, in_channels: 'int', num_of_vertices: 'int', num_of_timesteps: 'int'):
        super(SpatialAttention, self).__init__()
        self._W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self._W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self._W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self._bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self._Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: 'torch.FloatTensor') ->torch.FloatTensor:
        """
        Making a forward pass of the spatial attention layer.
        
        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **S** (PyTorch FloatTensor) - Spatial attention score matrices, with shape (B, N_nodes, N_nodes).
        """
        LHS = torch.matmul(torch.matmul(X, self._W1), self._W2)
        RHS = torch.matmul(self._W3, X).transpose(-1, -2)
        S = torch.matmul(self._Vs, torch.sigmoid(torch.matmul(LHS, RHS) + self._bs))
        S = F.softmax(S, dim=1)
        return S


class TemporalAttention(nn.Module):
    """An implementation of the Temporal Attention Module. For details see this paper: 
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow 
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, in_channels: 'int', num_of_vertices: 'int', num_of_timesteps: 'int'):
        super(TemporalAttention, self).__init__()
        self._U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))
        self._U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices))
        self._U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self._be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps))
        self._Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: 'torch.FloatTensor') ->torch.FloatTensor:
        """
        Making a forward pass of the temporal attention layer.
       
        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **E** (PyTorch FloatTensor) - Temporal attention score matrices, with shape (B, T_in, T_in).
        """
        LHS = torch.matmul(torch.matmul(X.permute(0, 3, 2, 1), self._U1), self._U2)
        RHS = torch.matmul(self._U3, X)
        E = torch.matmul(self._Ve, torch.sigmoid(torch.matmul(LHS, RHS) + self._be))
        E = F.softmax(E, dim=1)
        return E


class ASTGCNBlock(nn.Module):
    """An implementation of the Attention Based Spatial-Temporal Graph Convolutional Block.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional 
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels: 'int', K: 'int', nb_chev_filter: 'int', nb_time_filter: 'int', time_strides: 'int', num_of_vertices: 'int', num_of_timesteps: 'int', normalization: 'Optional[str]'=None, bias: 'bool'=True):
        super(ASTGCNBlock, self).__init__()
        self._temporal_attention = TemporalAttention(in_channels, num_of_vertices, num_of_timesteps)
        self._spatial_attention = SpatialAttention(in_channels, num_of_vertices, num_of_timesteps)
        self._chebconv_attention = ChebConvAttention(in_channels, nb_chev_filter, K, normalization, bias)
        self._time_convolution = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self._residual_convolution = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self._normalization = normalization
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: 'torch.FloatTensor', edge_index: 'Union[torch.LongTensor, List[torch.LongTensor]]') ->torch.FloatTensor:
        """
        Making a forward pass with the ASTGCN block.
 
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * **edge_index** (LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = X.shape
        X_tilde = self._temporal_attention(X)
        X_tilde = torch.matmul(X.reshape(batch_size, -1, num_of_timesteps), X_tilde)
        X_tilde = X_tilde.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        X_tilde = self._spatial_attention(X_tilde)
        if not isinstance(edge_index, list):
            data = Data(edge_index=edge_index, edge_attr=None, num_nodes=num_of_vertices)
            if self._normalization != 'sym':
                lambda_max = LaplacianLambdaMax()(data).lambda_max
            else:
                lambda_max = None
            X_hat = []
            for t in range(num_of_timesteps):
                X_hat.append(torch.unsqueeze(self._chebconv_attention(X[:, :, :, t], edge_index, X_tilde, lambda_max=lambda_max), -1))
            X_hat = F.relu(torch.cat(X_hat, dim=-1))
        else:
            X_hat = []
            for t in range(num_of_timesteps):
                data = Data(edge_index=edge_index[t], edge_attr=None, num_nodes=num_of_vertices)
                if self._normalization != 'sym':
                    lambda_max = LaplacianLambdaMax()(data).lambda_max
                else:
                    lambda_max = None
                X_hat.append(torch.unsqueeze(self._chebconv_attention(X[:, :, :, t], edge_index[t], X_tilde, lambda_max=lambda_max), -1))
            X_hat = F.relu(torch.cat(X_hat, dim=-1))
        X_hat = self._time_convolution(X_hat.permute(0, 2, 1, 3))
        X = self._residual_convolution(X.permute(0, 2, 1, 3))
        X = self._layer_norm(F.relu(X + X_hat).permute(0, 3, 2, 1))
        X = X.permute(0, 2, 3, 1)
        return X


class ASTGCN(nn.Module):
    """An implementation of the Attention Based Spatial-Temporal Graph Convolutional Cell.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional 
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        edge_index (array): edge indices.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
        num_of_vertices (int): Number of vertices in the graph.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, nb_block: 'int', input_dim: 'int', output_dim: 'int', max_view: 'int', nb_chev_filter: 'int', nb_time_filter: 'int', time_strides: 'int', horizon: 'int', seq_len: 'int', node_num: 'int', normalization: 'Optional[str]'=None, bias: 'bool'=True, **model_kwargs):
        super(ASTGCN, self).__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self._blocklist = nn.ModuleList([ASTGCNBlock(input_dim, max_view, nb_chev_filter, nb_time_filter, time_strides, node_num, seq_len, normalization, bias)])
        self._blocklist.extend([ASTGCNBlock(nb_time_filter, max_view, nb_chev_filter, nb_time_filter, 1, node_num, seq_len // time_strides, normalization, bias) for _ in range(nb_block - 1)])
        self._final_conv = nn.Conv2d(int(seq_len / time_strides), horizon * output_dim, kernel_size=(1, nb_time_filter))
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: 'torch.FloatTensor', edge_index: 'torch.LongTensor', *args) ->torch.FloatTensor:
        """
        Making a forward pass.
        
        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * **edge_index** (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * **X** (PyTorch FloatTensor)* - Hidden state tensor for all nodes, with shape (B, N_nodes, T_out).
        """
        B, N_nodes, F_in, T_in = X.shape
        F_out = self.output_dim
        T_out = self.horizon
        for block in self._blocklist:
            X = block(X, edge_index)
        X = self._final_conv(X.permute(0, 3, 1, 2))
        X = X[:, :, :, -1]
        X = X.reshape(B, T_out, F_out, N_nodes)
        X = X.permute(1, 0, 3, 2)
        return X


class Seq2SeqAttrs:

    def __init__(self, sparse_idx, angle_ratio, geodesic, **model_kwargs):
        self.sparse_idx = sparse_idx
        self.max_view = int(model_kwargs.get('max_view', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.node_num = int(model_kwargs.get('node_num', 6))
        self.layer_num = int(model_kwargs.get('layer_num', 2))
        self.rnn_units = int(model_kwargs.get('rnn_units', 32))
        self.input_dim = int(model_kwargs.get('input_dim', 2))
        self.output_dim = int(model_kwargs.get('output_dim', 2))
        self.seq_len = int(model_kwargs.get('seq_len', 12))
        self.lck_structure = model_kwargs.get('lckstructure', [4, 8])
        self.embed_dim = int(model_kwargs.get('embed_dim', 16))
        self.location_dim = int(model_kwargs.get('location_dim', 16))
        self.horizon = int(model_kwargs.get('horizon', 16))
        self.hidden_units = int(model_kwargs.get('hidden_units', 16))
        self.block_num = int(model_kwargs.get('block_num', 2))
        angle_ratio = torch.sparse.FloatTensor(self.sparse_idx, angle_ratio, (self.node_num, self.node_num)).to_dense()
        self.angle_ratio = angle_ratio + torch.eye(*angle_ratio.shape)
        self.geodesic = torch.sparse.FloatTensor(self.sparse_idx, geodesic, (self.node_num, self.node_num)).to_dense()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ATTModel(nn.Module, Seq2SeqAttrs):

    def __init__(self, sparse_idx, attention_method, logger=None, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, **model_kwargs)
        if attention_method in ['ASTGCN', 'MSTGCN']:
            self.network = getattr(attention, attention_method)(nb_block=self.block_num, nb_chev_filter=self.hidden_units, nb_time_filter=self.hidden_units, time_strides=int(self.seq_len / 2), **model_kwargs)
        elif attention_method in ['STGCN']:
            self.network = getattr(attention, attention_method)(kernel_size=int(self.seq_len / 2), **model_kwargs)
        self._logger = logger

    def forward(self, inputs, labels=None, batches_seen=None):
        inputs = inputs.permute(1, 2, 3, 0)
        outputs = self.network(inputs, self.sparse_idx)
        if batches_seen == 0:
            self._logger.info('Total trainable parameters {}'.format(count_parameters(self)))
        return outputs


class GraphConv(nn.Module):

    def __init__(self, input_dim, output_dim, max_view, conv):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = conv
        self.max_view = max_view
        self.linear = nn.Linear(input_dim * max_view, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class MSTGCNBlock(nn.Module):
    """An implementation of the Multi-Component Spatial-Temporal Graph
    Convolution block `_
    
    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
    """

    def __init__(self, in_channels: 'int', conv_ker, max_view: 'int', nb_chev_filter: 'int', nb_time_filter: 'int', time_strides: 'int'):
        super(MSTGCNBlock, self).__init__()
        self.conv_ker = conv_ker
        self.conv = GraphConv(input_dim=in_channels, output_dim=nb_chev_filter, max_view=max_view, conv=conv_ker)
        self._time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self._residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self.nb_time_filter = nb_time_filter
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: 'torch.FloatTensor') ->torch.FloatTensor:
        """
        Making a forward pass with a single MSTGCN block.

        Arg types:
            * X (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in). 
            * edge_index (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * X (PyTorch FloatTensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """
        num_of_timesteps, batch_size, num_of_vertices, in_channels = X.shape
        X_tilde = X
        X_tilde = F.relu(self.conv(x=X_tilde))
        X_tilde = X_tilde.permute(1, 3, 2, 0)
        X_tilde = self._time_conv(X_tilde)
        X = self._residual_conv(X.permute(1, 3, 2, 0))
        X = self._layer_norm(F.relu(X + X_tilde).permute(0, 3, 2, 1))
        X = X.permute(1, 0, 2, 3)
        return X


class MSTGCN(nn.Module):
    """An implementation of the Multi-Component Spatial-Temporal Graph Convolution Networks, a degraded version of ASTGCN.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional 
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_
    
    Args:
        
        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
    """

    def __init__(self, nb_block: 'int', input_dim: 'int', output_dim: 'int', max_view: 'int', nb_chev_filter: 'int', nb_time_filter: 'int', time_strides: 'int', seq_len: 'int', horizon: 'int', **model_kwargs):
        super(MSTGCN, self).__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self._blocklist = nn.ModuleList([MSTGCNBlock(input_dim, max_view, nb_chev_filter, nb_time_filter, time_strides)])
        self._blocklist.extend([MSTGCNBlock(nb_time_filter, max_view, nb_chev_filter, nb_time_filter, 1) for _ in range(nb_block - 1)])
        self._final_conv = nn.Conv2d(int(seq_len / time_strides), seq_len * output_dim, kernel_size=(1, nb_time_filter))
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: 'torch.FloatTensor', edge_index: 'torch.LongTensor') ->torch.FloatTensor:
        """ Making a forward pass. This module takes a likst of MSTGCN blocks and use a final convolution to serve as a multi-component fusion.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features. 
        T_in is the length of input sequence in time. T_out is the length of output sequence in time.
        
        Arg types:
            * X (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * edge_index (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * X (PyTorch FloatTensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, T_out).
        """
        B, N_nodes, F_in, T_in = X.shape
        F_out = self.output_dim
        T_out = self.horizon
        for block in self._blocklist:
            X = block(X, edge_index)
        X = self._final_conv(X.permute(0, 3, 1, 2))
        X = X[:, :, :, -1]
        X = X.reshape(B, T_out, F_out, N_nodes)
        X = X.permute(1, 0, 3, 2)
        return X


class TemporalConv(nn.Module):
    """Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting." 
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int'=3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X: 'torch.FloatTensor') ->torch.FloatTensor:
        """Forward pass through temporal convolution block.
        
        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape 
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape 
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H


class STGCN(nn.Module):
    """Spatio-temporal convolution block using ChebConv Graph Convolutions. 
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting" 
    <https://arxiv.org/abs/1709.04875>`_

    NB. The ST-Conv block contains two temporal convolutions (TemporalConv) 
    with kernel size k. Hence for an input sequence of length m, 
    the output sequence will be length m-2(k-1).

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units output by graph convolution block
        out_channels (int): Number of output features.
        kernel_size (int): Size of the kernel considered. 
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(self, node_num: 'int', input_dim: 'int', hidden_units: 'int', output_dim: 'int', kernel_size: 'int', max_view: 'int', horizon: 'int', normalization: 'str'='sym', bias: 'bool'=True, **model_kwargs):
        super(STGCN, self).__init__()
        self.num_nodes = node_num
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.out_channels = output_dim
        self.kernel_size = kernel_size
        self.K = max_view
        self.normalization = normalization
        self.bias = bias
        self._temporal_conv1 = TemporalConv(in_channels=input_dim, out_channels=hidden_units, kernel_size=kernel_size)
        self._graph_conv = ChebConv(in_channels=hidden_units, out_channels=hidden_units, K=max_view, normalization=normalization, bias=bias)
        self._temporal_conv2 = TemporalConv(in_channels=hidden_units, out_channels=output_dim * horizon, kernel_size=kernel_size)
        self._batch_norm = nn.BatchNorm2d(node_num)

    def forward(self, X: 'torch.FloatTensor', edge_index: 'torch.LongTensor', edge_weight: 'torch.FloatTensor'=None, **kwargs) ->torch.FloatTensor:
        """Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. 

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.
        
        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        X = X.permute(0, 3, 1, 2)
        batch_size, seq_len, node_num, input_dim = X.shape
        T_0 = self._temporal_conv1(X)
        T = torch.zeros_like(T_0)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)
        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        T = self._batch_norm(T)
        T = T.permute(3, 0, 1, 2)[..., -1].reshape(seq_len, batch_size, node_num, -1)
        return T


class AVWGCN(nn.Module):
    """An implementation of the Node Adaptive Graph Convolution Layer.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', K: 'int', embedding_dimensions: 'int'):
        super(AVWGCN, self).__init__()
        self.K = K
        self.weights_pool = torch.nn.Parameter(torch.Tensor(embedding_dimensions, K, in_channels, out_channels))
        self.bias_pool = torch.nn.Parameter(torch.Tensor(embedding_dimensions, out_channels))
        glorot(self.weights_pool)
        zeros(self.bias_pool)

    def forward(self, X: 'torch.FloatTensor', E: 'torch.FloatTensor') ->torch.FloatTensor:
        """Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **E** (PyTorch Float Tensor) - Node embeddings.
        Return types:
            * **X_G** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        number_of_nodes = E.shape[0]
        supports = F.softmax(F.relu(torch.mm(E, E.transpose(0, 1))), dim=1)
        support_set = [torch.eye(number_of_nodes), supports]
        for _ in range(2, self.K):
            support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(support)
        supports = torch.stack(support_set, dim=0)
        W = torch.einsum('nd,dkio->nkio', E, self.weights_pool)
        bias = torch.matmul(E, self.bias_pool)
        X_G = torch.einsum('knm,bmc->bknc', supports, X)
        X_G = X_G.permute(0, 2, 1, 3)
        X_G = torch.einsum('bnki,nkio->bno', X_G, W) + bias
        return X_G


class AGCRN(nn.Module):
    """An implementation of the Adaptive Graph Convolutional Recurrent Unit.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        number_of_nodes (int): Number of vertices.
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(self, node_num: 'int', in_channels: 'int', out_channels: 'int', max_view: 'int', embed_dim: 'int', **model_kwargs):
        super(AGCRN, self).__init__()
        self.number_of_nodes = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = max_view
        self.embedding_dimensions = embed_dim
        self._setup_layers()
        self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_dim), requires_grad=True)

    def _setup_layers(self):
        self._gate = AVWGCN(in_channels=self.in_channels + self.out_channels, out_channels=2 * self.out_channels, K=self.K, embedding_dimensions=self.embedding_dimensions)
        self._update = AVWGCN(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, K=self.K, embedding_dimensions=self.embedding_dimensions)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels)
        return H

    def forward(self, X: 'torch.FloatTensor', H: 'torch.FloatTensor'=None, E: 'torch.FloatTensor'=None, **args) ->torch.FloatTensor:
        """Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node feature matrix.
            * **H** (PyTorch Float Tensor) - Node hidden state matrix. Default is None.
            * **E** (PyTorch Float Tensor) - Node embedding matrix.
        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        if E is None:
            E = self.node_embeddings
        H = self._set_hidden_state(X, H)
        X_H = torch.cat((X, H), dim=-1)
        Z_R = torch.sigmoid(self._gate(X_H, E))
        Z, R = torch.split(Z_R, self.out_channels, dim=-1)
        C = torch.cat((X, Z * H), dim=-1)
        HC = torch.tanh(self._update(C, E))
        H = R * H + (1 - R) * HC
        return H


class DCRNN(torch.nn.Module):
    """An implementation of the Diffusion Convolutional Gated Recurrent Unit.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): NUmber of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer 
            will not learn an additive bias (default :obj:`True`)

    """

    def __init__(self, in_channels: 'int', out_channels: 'int', max_view: 'int', bias: 'bool'=True, **model_kwargs):
        super(DCRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = max_view
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = DConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, K=self.K, bias=self.bias)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = DConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, K=self.K, bias=self.bias)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = DConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, K=self.K, bias=self.bias)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([X, H], dim=-1)
        Z = self.conv_x_z(Z, edge_index, edge_weight)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([X, H], dim=-1)
        R = self.conv_x_r(R, edge_index, edge_weight)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([X, H * R], dim=-1)
        H_tilde = self.conv_x_h(H_tilde, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X: 'torch.FloatTensor', edge_index: 'torch.LongTensor', edge_weight: 'torch.FloatTensor'=None, H: 'torch.FloatTensor'=None, **args) ->torch.FloatTensor:
        """Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.
            * **H** (PyTorch Float Tensor, optional) - Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class CLCRNCell(torch.nn.Module):

    def __init__(self, num_units, sparse_idx, max_view, node_num, num_feature, conv_ker, num_embedding, nonlinearity='tanh'):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._node_num = node_num
        self._num_feature = num_feature
        self._num_units = num_units
        self._max_view = max_view
        self._sparse_idx = sparse_idx
        self._num_embedding = num_embedding
        self.conv_ker = conv_ker
        self.ru_gconv = GraphConv(input_dim=self._num_embedding, output_dim=self._num_units * 2, max_view=self._max_view, conv=conv_ker)
        self.c_gconv = GraphConv(input_dim=self._num_embedding, output_dim=self._num_units, max_view=self._max_view, conv=conv_ker)

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, node_num, input_dim) 
        :param hx: (B, node_num, rnn_units)
        :param t: (B, num_time_feature)
        :return
        - Output: A `3-D` tensor with shaconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerconv=conv_kerpe `(B, node_num, rnn_units)`.
        """
        conv_in_ru = self._concat(inputs, hx)
        value = torch.sigmoid(self.ru_gconv(conv_in_ru))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._node_num, self._num_units))
        u = torch.reshape(u, (-1, self._node_num, self._num_units))
        conv_in_c = self._concat(inputs, r * hx)
        c = self.c_gconv(conv_in_c)
        if self._activation is not None:
            c = self._activation(c)
        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        return torch.cat([x, x_], dim=2)


class DecoderModel(nn.Module, Seq2SeqAttrs):

    def __init__(self, sparse_idx, geodesic, angle_ratio, conv_ker, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, sparse_idx, geodesic, angle_ratio, **model_kwargs)
        self.conv = conv_ker
        self.clgru_layers = self.init_clgru_layers()
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)

    def init_clgru_layers(self):
        module_list = []
        for i in range(self.layer_num):
            if i == 0:
                input_dim = self.output_dim
            else:
                input_dim = self.rnn_units
            module_list.append(CLCRNCell(self.rnn_units, self.sparse_idx, self.max_view, self.node_num, input_dim, self.conv, input_dim + self.rnn_units))
        return nn.ModuleList(module_list)

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.node_num * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.node_num * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.clgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output)
        output = projected.reshape(-1, self.node_num, self.output_dim)
        return output, torch.stack(hidden_states)


class EncoderModel(nn.Module, Seq2SeqAttrs):

    def __init__(self, sparse_idx, angle_ratio, geodesic, conv_ker, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, sparse_idx, angle_ratio, geodesic, **model_kwargs)
        self.conv = conv_ker
        self.clgru_layers = self.init_clgru_layers()
        self.projection_layer = nn.Linear(self.input_dim + 2 * self.embed_dim, self.rnn_units)

    def init_clgru_layers(self):
        module_list = []
        for i in range(self.layer_num):
            if i == 0:
                input_dim = self.input_dim
            else:
                input_dim = self.rnn_units
            module_list.append(CLCRNCell(self.rnn_units, self.sparse_idx, self.max_view, self.node_num, input_dim, self.conv, self.rnn_units * 2))
        return nn.ModuleList(module_list)

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.node_num * self.input_dim)
        :param hidden_state: (layer_num, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, __, _ = inputs.size()
        inputs = self.projection_layer(inputs)
        if hidden_state is None:
            hidden_state = torch.zeros((self.layer_num, batch_size, self.node_num, self.rnn_units))
            hidden_state = hidden_state
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.clgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class GConvGRU(torch.nn.Module):
    """An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', max_view: 'int', normalization: 'str'='sym', bias: 'bool'=True, **model_kwargs):
        super(GConvGRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = max_view
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_z = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_r = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_h = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = self.conv_x_z(X, edge_index, edge_weight)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = self.conv_x_r(X, edge_index, edge_weight)
        R = R + self.conv_h_r(H, edge_index, edge_weight)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X: 'torch.FloatTensor', edge_index: 'torch.LongTensor', edge_weight: 'torch.FloatTensor'=None, H: 'torch.FloatTensor'=None, **args) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class RNNModel(nn.Module, Seq2SeqAttrs):

    def __init__(self, sparse_idx, conv_method, logger=None, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, **model_kwargs)
        conv = []
        for i in range(self.layer_num):
            if i == 0:
                conv.append(getattr(recurrent, conv_method)(in_channels=self.input_dim, out_channels=self.rnn_units, **model_kwargs))
            else:
                conv.append(getattr(recurrent, conv_method)(in_channels=self.rnn_units, out_channels=self.rnn_units, K=self.max_view, **model_kwargs))
        self._logger = logger
        self.conv = nn.ModuleList(conv)
        self.encoder_model = EncoderModel(sparse_idx, self.conv, **model_kwargs)
        self.decoder_model = DecoderModel(sparse_idx, self.conv, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.node_num, self.output_dim))
        go_symbol = go_symbol
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []
        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)
        :param labels: shape (horizon, batch_size, num_sensor, output_dim)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.node_num * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        if batches_seen == 0:
            self._logger.info('Total trainable parameters {}'.format(count_parameters(self)))
        return outputs


class TGCN(torch.nn.Module):
    """An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is True.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', improved: 'bool'=False, cached: 'bool'=False, add_self_loops: 'bool'=True, **model_args):
        super(TGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], dim=-1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], dim=-1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], dim=-1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X: 'torch.FloatTensor', edge_index: 'torch.LongTensor', edge_weight: 'torch.FloatTensor'=None, H: 'torch.FloatTensor'=None, **args) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class CLCSTN(nn.Module):
    """An implementation of the Multi-Component Spatial-Temporal Graph Convolution Networks, a degraded version of ASTGCN.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional 
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_
    
    Args:
        
        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
    """

    def __init__(self, nb_block: 'int', input_dim: 'int', output_dim: 'int', max_view: 'int', nb_chev_filter: 'int', nb_time_filter: 'int', time_strides: 'int', seq_len: 'int', horizon: 'int', conv_ker, embed_dim=None, **model_kwargs):
        super(CLCSTN, self).__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.conv_ker = conv_ker
        if embed_dim is not None:
            self.input_dim = input_dim + embed_dim * 2
        self._blocklist = nn.ModuleList([MSTGCNBlock(self.input_dim, self.conv_ker, max_view, nb_chev_filter, nb_time_filter, time_strides)])
        self._blocklist.extend([MSTGCNBlock(nb_time_filter, self.conv_ker, max_view, nb_chev_filter, nb_time_filter, 1) for _ in range(nb_block - 1)])
        self._final_conv = nn.Conv2d(int(seq_len / time_strides), seq_len * output_dim, kernel_size=(1, nb_time_filter))
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: 'torch.FloatTensor') ->torch.FloatTensor:
        T_in, B, N_nodes, F_in = X.shape
        F_out = self.output_dim
        T_out = self.horizon
        for block in self._blocklist:
            X = block(X)
        X = self._final_conv(X.permute(1, 0, 2, 3))
        X = X[:, :, :, -1]
        X = X.reshape(B, T_out, F_out, N_nodes)
        X = X.permute(1, 0, 3, 2)
        return X


class Attention(nn.Module):

    def __init__(self, location_dim, embed_dim=8):
        super(Attention, self).__init__()
        self.query_linear = nn.Linear(location_dim, embed_dim)
        self.key_linear = nn.Linear(location_dim, embed_dim)

    def forward(self, q, k):
        """
        q: b, l, n, f
        k: b, l, n, f
        """
        q = self.query_linear(q)
        k = self.key_linear(k)
        att = torch.matmul(q, k.transpose(-2, -1))
        return att.squeeze()


class ConvAttrs:

    def __init__(self, location_dim, sparse_idx, node_num, lck_structure, local_graph_coor, angle_ratio, geodesic, max_view):
        self.location_dim = location_dim
        self.node_num = node_num
        self.sparse_idx = sparse_idx
        self.local_graph_coor = local_graph_coor
        self.angle_ratio = angle_ratio
        self.geodesic = geodesic
        self.max_view = max_view
        self.lck_structure = lck_structure


class LocalConditionalKer(nn.Module):

    def __init__(self, location_dim, structure, activation='tanh'):
        super(LocalConditionalKer, self).__init__()
        """Initialize the generator.
        """
        self.structure = structure
        self.network = nn.ModuleList()
        self.network.append(nn.Linear(location_dim * 2, self.structure[0]))
        for i in range(len(self.structure) - 1):
            self.network.append(nn.Linear(self.structure[i], self.structure[i + 1]))
            self.network.append(nn.BatchNorm1d(self.structure[i + 1]))
        self.network.append(nn.Linear(self.structure[-1], 1))
        if activation == 'tanh':
            self.activation = torch.tanh

    def forward(self, x):
        for j, layer in enumerate(self.network):
            if j != 1:
                x = self.activation(x)
            x = layer(x)
        return torch.relu(x)


class CLConv(nn.Module, ConvAttrs):

    def __init__(self, location_dim, sparse_idx, node_num, lck_structure, local_graph_coor, angle_ratio, geodesic, max_view):
        super(CLConv, self).__init__()
        ConvAttrs.__init__(self, location_dim, sparse_idx, node_num, lck_structure, local_graph_coor, angle_ratio, geodesic, max_view)
        assert local_graph_coor.shape[-1] == location_dim * 2
        self.lcker = LocalConditionalKer(location_dim, lck_structure)
        self.weight_att = Attention(location_dim)

    def conv_kernel(self, coor):
        lcker = self.lcker(coor)
        lcker = torch.sparse.FloatTensor(self.sparse_idx, lcker.flatten(), (self.node_num, self.node_num)).to_dense()
        sphere_coor = coor[:, :self.location_dim] + coor[:, self.location_dim:]
        sphere_coor = sphere_coor.reshape(self.node_num, -1, self.location_dim)
        center_points = sphere_coor[:, [0], :]
        neighbor_points = sphere_coor
        alpha = self.weight_att(center_points, neighbor_points).abs()
        alpha = torch.sparse.FloatTensor(self.sparse_idx, alpha.flatten(), (self.node_num, self.node_num)).to_dense()
        distance_decay = (-alpha * self.geodesic).exp()
        angle_ratio = self.angle_ratio
        return lcker * distance_decay * angle_ratio

    def forward(self, x):
        kernel = self.conv_kernel(self.local_graph_coor)
        x = torch.cat([torch.matmul(torch.pow(kernel, i + 1), x) for i in range(self.max_view)], dim=-1)
        return x

    def kernel_prattern(self, centers, vs, angle_ratio):
        """
        centers: (N, 2)
        vs: (N, M, 2)
        """
        M = vs.shape[1]
        centers = centers[:, None, :].repeat(1, M, 1)
        coor = torch.cat([centers, vs], dim=-1)
        ker_patterns = []
        for i in range(coor.shape[0]):
            lcker = self.lcker(coor[i]).flatten()
            geodesics = torch.square(vs[i]).sum(dim=-1).sqrt()
            alpha = self.weight_att(centers[i], vs[i]).abs()
            distance_decay = (-alpha[0] * geodesics).exp()
            ker_patterns.append(lcker * distance_decay * angle_ratio)
        return torch.stack(ker_patterns, dim=0)


class CLCSTNModel(nn.Module, Seq2SeqAttrs):

    def __init__(self, loc_info, sparse_idx, geodesic, angle_ratio, logger=None, **model_kwargs):
        """
        Conditional Local Convolution Recurrent Network, implemented based on DCRNN,
        Args:
            loc_info (torch.Tensor): location infomation of each nodes, with the shape (node_num, location_dim). For sphercial signals, location_dim=2.
            sparse_idx (torch.Tensor): sparse_idx with the shape (2, node_num * nbhd_num).
            geodesic (torch.Tensor): geodesic distance between each point and its neighbors, with the shape (node_num * nbhd_num), corresponding to sparse_idx.
            angle_ratio (torch.Tensor): the defined angle ratio contributing to orientation density, with the shape (node_num * nbhd_num), corresponding to sparse_idx.
            model_kwargs (dict): Other model args see the config.yaml.
        """
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, geodesic, angle_ratio, **model_kwargs)
        self.register_buffer('node_embeddings', nn.Parameter(torch.randn(self.node_num, self.embed_dim), requires_grad=True))
        self.feature_embedding = nn.Linear(self.input_dim, self.embed_dim)
        self._logger = logger
        self.conv_ker = CLConv(self.location_dim, self.sparse_idx, self.node_num, self.lck_structure, loc_info, self.angle_ratio, self.geodesic, self.max_view)
        self.network = CLCSTN(nb_block=self.block_num, nb_chev_filter=self.hidden_units, nb_time_filter=self.hidden_units, time_strides=int(self.seq_len / 2), conv_ker=self.conv_ker, **model_kwargs)

    def embedding(self, inputs):
        batch_size, seq_len, node_num, feature_size = inputs.shape
        feature_emb = self.feature_embedding(inputs)
        node_emb = self.node_embeddings[None, None, :, :].expand(batch_size, seq_len, node_num, self.embed_dim)
        return torch.cat([feature_emb, node_emb, inputs], dim=-1)

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)
        :param labels: shape (horizon, batch_size, num_sensor, output_dim)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.node_num * self.output_dim)
        """
        embedding = self.embedding(inputs)
        outputs = self.network(embedding)
        if batches_seen == 0:
            self._logger.info('Total trainable parameters {}'.format(count_parameters(self)))
        return outputs


class CLCRNModel(nn.Module, Seq2SeqAttrs):

    def __init__(self, loc_info, sparse_idx, geodesic, angle_ratio, logger=None, **model_kwargs):
        """
        Conditional Local Convolution Recurrent Network, implemented based on DCRNN,
        Args:
            loc_info (torch.Tensor): location infomation of each nodes, with the shape (node_num, location_dim). For sphercial signals, location_dim=2.
            sparse_idx (torch.Tensor): sparse_idx with the shape (2, node_num * nbhd_num).
            geodesic (torch.Tensor): geodesic distance between each point and its neighbors, with the shape (node_num * nbhd_num), corresponding to sparse_idx.
            angle_ratio (torch.Tensor): the defined angle ratio contributing to orientation density, with the shape (node_num * nbhd_num), corresponding to sparse_idx.
            model_kwargs (dict): Other model args see the config.yaml.
        """
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, geodesic, angle_ratio, **model_kwargs)
        self.register_buffer('node_embeddings', nn.Parameter(torch.randn(self.node_num, self.embed_dim), requires_grad=True))
        self.feature_embedding = nn.Linear(self.input_dim, self.embed_dim)
        self.conv_ker = CLConv(self.location_dim, self.sparse_idx, self.node_num, self.lck_structure, loc_info, self.angle_ratio, self.geodesic, self.max_view)
        self.encoder_model = EncoderModel(sparse_idx, geodesic, angle_ratio, self.conv_ker, **model_kwargs)
        self.decoder_model = DecoderModel(sparse_idx, geodesic, angle_ratio, self.conv_ker, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def get_kernel(self):
        return self.conv_ker

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.node_num, self.output_dim))
        go_symbol = go_symbol
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []
        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def embedding(self, inputs):
        batch_size, seq_len, node_num, feature_size = inputs.shape
        feature_emb = self.feature_embedding(inputs)
        node_emb = self.node_embeddings[None, None, :, :].expand(batch_size, seq_len, node_num, self.embed_dim)
        return torch.cat([feature_emb, node_emb, inputs], dim=-1)

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)
        :param labels: shape (horizon, batch_size, num_sensor, output_dim)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.node_num * self.output_dim)
        """
        embedding = self.embedding(inputs)
        encoder_hidden_state = self.encoder(embedding)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        if batches_seen == 0:
            self._logger.info('Total trainable parameters {}'.format(count_parameters(self)))
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'location_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialAttention,
     lambda: ([], {'in_channels': 4, 'num_of_vertices': 4, 'num_of_timesteps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TemporalAttention,
     lambda: ([], {'in_channels': 4, 'num_of_vertices': 4, 'num_of_timesteps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TemporalConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_EDAPINENUT_CLCRN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

