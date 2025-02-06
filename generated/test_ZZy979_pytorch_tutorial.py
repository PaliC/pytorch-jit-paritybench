import sys
_module = sys.modules[__name__]
del sys
beginner = _module
cifar10_tutorial = _module
neural_networks_tutorial = _module
two_layer_net = _module
dlwizard = _module
cnn = _module
common = _module
fnn = _module
linear_regression = _module
logistic_regression = _module
lstm = _module
rnn = _module
gnn = _module
cs = _module
model = _module
train = _module
data = _module
acm = _module
aminer = _module
dblp = _module
dgl = _module
heco = _module
imdb = _module
dgl_first_demo = _module
edge_clf = _module
edge_clf_hetero = _module
edge_clf_hetero_mb = _module
edge_clf_mb = _module
edge_type_hetero = _module
graph_clf = _module
graph_clf_hetero = _module
link_pred = _module
link_pred_hetero = _module
link_pred_hetero_mb = _module
link_pred_mb = _module
model = _module
node_clf = _module
node_clf_hetero = _module
node_clf_hetero_mb = _module
node_clf_mb = _module
gat = _module
model = _module
train_inductive = _module
train_transductive = _module
gcn = _module
model = _module
train = _module
han = _module
model = _module
train = _module
model = _module
train = _module
hetgnn = _module
eval = _module
model = _module
preprocess = _module
random_walk = _module
train = _module
utils = _module
hgconv = _module
model = _module
train = _module
hgt = _module
model = _module
train = _module
lp = _module
model = _module
magnn = _module
encoder = _module
model = _module
train_dblp = _module
train_imdb = _module
metapath2vec = _module
random_walk = _module
skipgram = _module
train = _module
train_word2vec = _module
rgcn = _module
model = _module
model_hetero = _module
train_entity_clf = _module
train_link_pred = _module
rhgnn = _module
model = _module
train = _module
sign = _module
model = _module
train = _module
supergat = _module
attention = _module
model = _module
train = _module
utils = _module
metapath = _module
metrics = _module
neg_sampler = _module
random_walk = _module
kgrec = _module
kgcn = _module
data = _module
dataloader = _module
model = _module
train = _module
nlp = _module
tfms = _module
causal_lm_model = _module
eqa_model = _module
eqa_pipeline = _module
masked_lm_model = _module
masked_lm_pipeline = _module
seq_clf_model = _module
seq_clf_pipeline = _module
text_gen_model = _module
text_gen_pipeline = _module

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


import matplotlib.pyplot as plt


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torchvision


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import DataLoader


from torchvision.datasets import MNIST


import scipy.io as sio


import pandas as pd


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords


import scipy.sparse as sp


import itertools


from sklearn.feature_extraction.text import CountVectorizer


import matplotlib.animation as animation


import random


from sklearn.metrics import f1_score


from sklearn.cluster import KMeans


from sklearn.metrics import roc_auc_score


from sklearn.metrics import normalized_mutual_info_score


from sklearn.metrics import adjusted_rand_score


from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split


from collections import Counter


from itertools import chain


import warnings


import math


from collections import defaultdict


from sklearn.preprocessing import LabelEncoder


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class TwoLayerNet(nn.Module):
    """包含两个全连接层的前馈神经网络"""

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as member variables.
        """
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.linear2(F.relu(self.linear1(x)))


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        """
        :param x: tensor(*, 28, 28) 输入图像
        :return: tensor(*, 10)
        """
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        return out


class FeedforwardNeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class LSTM(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        :param x: tensor(batch, seq_len, d_in)
        :return: tensor(batch, d_out)
        """
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class RNN(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(in_dim, hidden_dim, num_layers, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        :param x: tensor(batch, seq_len, d_in)
        :return: tensor(batch, d_out)
        """
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.0):
        super().__init__()
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = F.relu(x)
            x = self.batch_norms[i](x)
            x = self.dropout(x)
        x = self.linears[-1](x)
        return x


class LabelPropagation(nn.Module):

    def __init__(self, num_layers, alpha):
        """标签传播模型

        .. math::
            Y^{(t+1)} = \\alpha D^{-1/2}AD^{-1/2}Y^{(t)} + (1-\\alpha)Y^{(t)}

        :param num_layers: int 传播层数
        :param alpha: float α参数
        """
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, g, labels, mask=None):
        """
        :param g: DGLGraph 无向图
        :param labels: tensor(N) 标签
        :param mask: tensor(N), optional 有标签顶点mask
        :return: tensor(N, C) 预测标签概率
        """
        with g.local_scope():
            labels = F.one_hot(labels).float()
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]
            else:
                y = labels
            degs = g.in_degrees().clamp(min=1)
            norm = torch.pow(degs, -0.5).unsqueeze(1)
            for _ in range(self.num_layers):
                g.ndata['h'] = norm * y
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = self.alpha * norm * g.ndata.pop('h') + (1 - self.alpha) * y
            return y


class CorrectAndSmooth(nn.Module):

    def __init__(self, num_correct_layers, correct_alpha, correct_norm, num_smooth_layers, smooth_alpha, smooth_norm, scale=1.0):
        """C&S模型"""
        super().__init__()
        self.correct_prop = LabelPropagation(num_correct_layers, correct_alpha, correct_norm)
        self.smooth_prop = LabelPropagation(num_smooth_layers, smooth_alpha, smooth_norm)
        self.scale = scale

    def correct(self, g, labels, base_pred, mask):
        """Correct步，修正基础预测中的误差

        :param g: DGLGraph 无向图
        :param labels: tensor(N, C) one-hot标签
        :param base_pred: tensor(N, C) 基础预测
        :param mask: tensor(N) 训练集mask
        :return: tensor(N, C) 修正后的预测
        """
        err = torch.zeros_like(base_pred)
        err[mask] = labels[mask] - base_pred[mask]

        def fix_input(y):
            y[mask] = err[mask]
            return y
        smoothed_err = self.correct_prop(g, err, post_step=fix_input)
        corrected_pred = base_pred + self.scale * smoothed_err
        corrected_pred[corrected_pred.isnan()] = base_pred[corrected_pred.isnan()]
        return corrected_pred

    def smooth(self, g, labels, corrected_pred, mask):
        """Smooth步，平滑最终预测

        :param g: DGLGraph 无向图
        :param labels: tensor(N, C) one-hot标签
        :param corrected_pred: tensor(N, C) 修正后的预测
        :param mask: tensor(N) 训练集mask
        :return: tensor(N, C) 最终预测
        """
        guess = corrected_pred
        guess[mask] = labels[mask]
        return self.smooth_prop(g, guess)

    def forward(self, g, labels, base_pred, mask):
        """
        :param g: DGLGraph 无向图
        :param labels: tensor(N, C) one-hot标签
        :param base_pred: tensor(N, C) 基础预测
        :param mask: tensor(N) 训练集mask
        :return: tensor(N, C) 最终预测
        """
        corrected_pred = self.correct(g, labels, base_pred, mask)
        return self.smooth(g, labels, corrected_pred, mask)


class DotProductPredictor(nn.Module):

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class GCN(nn.Module):
    """两层GCN模型"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, x):
        """
        :param g: DGLGraph
        :param x: tensor(N, d_in) 输入顶点特征，N为g的顶点数
        :return: tensor(N, d_out) 输出顶点特征
        """
        h = self.conv1(g, x)
        h = self.conv2(g, h)
        return h


class Model(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn = GCN(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, pos_g, neg_g, blocks, x):
        h = self.gcn(blocks, x)
        return self.pred(pos_g, h), self.pred(neg_g, h)


class RGCNFull(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({rel: GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({rel: GraphConv(hid_feats, out_feats) for rel in rel_names}, aggregate='sum')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        return h


class HeteroClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCNFull(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = self.rgcn(g, g.ndata['feat'])
        with g.local_scope():
            g.ndata['h'] = h
            hg = sum(dgl.mean_nodes(g, 'h', ntype=ntype) for ntype in g.ntypes)
            return F.softmax(self.classify(hg), dim=0)


class GCNFull(nn.Module):

    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class SAGEFull(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hid_feats, 'mean')
        self.conv2 = SAGEConv(hid_feats, out_feats, 'mean')

    def forward(self, g, inputs):
        h = F.relu(self.conv1(g, inputs))
        h = self.conv2(g, h)
        return h


class RGCN(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({rel: GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({rel: GraphConv(hid_feats, out_feats) for rel in rel_names}, aggregate='sum')

    def forward(self, blocks, inputs):
        h = self.conv1(blocks[0], inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(blocks[1], h)
        return h


class MLPPredictor(nn.Module):

    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        score = self.W(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class HeteroDotProductPredictor(nn.Module):

    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class HeteroMLPPredictor(nn.Module):

    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        score = self.W(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
        return {'score': score}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']


class MarginLoss(nn.Module):

    def forward(self, pos_score, neg_score):
        return (1 - pos_score + neg_score.view(pos_score.shape[0], -1)).clamp(min=0).mean()


class GAT(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout, residual=False, activation=None):
        """GAT模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: List[int] 每一层的注意力头数，长度等于层数
        :param dropout: float Dropout概率
        :param residual: bool, optional 是否使用残差连接，默认为False
        :param activation: callable, optional 输出层激活函数
        :raise ValueError: 如果层数（即num_heads的长度）小于2
        """
        super().__init__()
        num_layers = len(num_heads)
        if num_layers < 2:
            raise ValueError('层数至少为2，实际为{}'.format(num_layers))
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, num_heads[0], dropout, dropout, residual=residual, activation=F.elu))
        for i in range(1, num_layers - 1):
            self.layers.append(GATConv(num_heads[i - 1] * hidden_dim, hidden_dim, num_heads[i], dropout, dropout, residual=residual, activation=F.elu))
        self.layers.append(GATConv(num_heads[-2] * hidden_dim, out_dim, num_heads[-1], dropout, dropout, residual=residual, activation=activation))

    def forward(self, g, h):
        """
        :param g: DGLGraph 同构图
        :param h: tensor(N, d_in) 输入特征，N为g的顶点数
        :return: tensor(N, d_out) 输出顶点特征，K为注意力头数
        """
        for i in range(len(self.layers) - 1):
            h = self.layers[i](g, h).flatten(start_dim=1)
        h = self.layers[-1](g, h).mean(dim=1)
        return h


class SemanticAttention(nn.Module):

    def __init__(self, in_dim, hidden_dim=128):
        """语义层次的注意力，将顶点基于不同元路径的嵌入组合为最终嵌入

        :param in_dim: 输入特征维数d_in，对应顶点层次注意力的输出维数
        :param hidden_dim: 语义层次隐含特征维数d_hid
        """
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, z):
        """
        :param z: tensor(N, M, d_in) 顶点基于不同元路径的嵌入，N为顶点数，M为元路径个数
        :return: tensor(N, d_in) 顶点的最终嵌入
        """
        w = self.project(z).mean(dim=0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        z = (beta * z).sum(dim=1)
        return z


class HANLayer(nn.Module):

    def __init__(self, num_metapaths, in_dim, out_dim, num_heads, dropout):
        """HAN层

        :param num_metapaths: int 元路径个数
        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param dropout: float Dropout概率
        """
        super().__init__()
        self.gats = nn.ModuleList([GATConv(in_dim, out_dim, num_heads, dropout, dropout, activation=F.elu) for _ in range(num_metapaths)])
        self.semantic_attention = SemanticAttention(in_dim=num_heads * out_dim)

    def forward(self, gs, h):
        """
        :param gs: List[DGLGraph] 基于元路径的邻居组成的同构图
        :param h: tensor(N, d_in) 输入顶点特征
        :return: tensor(N, K*d_out) 输出顶点特征
        """
        zp = [gat(g, h).flatten(start_dim=1) for gat, g in zip(self.gats, gs)]
        zp = torch.stack(zp, dim=1)
        z = self.semantic_attention(zp)
        return z


class HAN(nn.Module):

    def __init__(self, num_metapaths, in_dim, hidden_dim, out_dim, num_heads, dropout):
        """HAN模型

        :param num_metapaths: int 元路径个数
        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param dropout: float Dropout概率
        """
        super().__init__()
        self.han = HANLayer(num_metapaths, in_dim, hidden_dim, num_heads, dropout)
        self.predict = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, gs, h):
        """
        :param gs: List[DGLGraph] 基于元路径的邻居组成的同构图
        :param h: tensor(N, d_in) 输入顶点特征
        :return: tensor(N, d_out) 输出顶点嵌入
        """
        h = self.han(gs, h)
        out = self.predict(h)
        return out


class HeCoGATConv(nn.Module):

    def __init__(self, hidden_dim, attn_drop=0.0, negative_slope=0.01, activation=None):
        """HeCo作者代码中使用的GAT

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.01
        :param activation: callable, optional 激活函数，默认为None
        """
        super().__init__()
        self.attn_l = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain)
        nn.init.xavier_normal_(self.attn_r, gain)

    def forward(self, g, feat_src, feat_dst):
        """
        :param g: DGLGraph 邻居-目标顶点二分图
        :param feat_src: tensor(N_src, d) 邻居顶点输入特征
        :param feat_dst: tensor(N_dst, d) 目标顶点输入特征
        :return: tensor(N_dst, d) 目标顶点输出特征
        """
        with g.local_scope():
            attn_l = self.attn_drop(self.attn_l)
            attn_r = self.attn_drop(self.attn_r)
            el = (feat_src * attn_l).sum(dim=-1).unsqueeze(dim=-1)
            er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(dim=-1)
            g.srcdata.update({'ft': feat_src, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = edge_softmax(g, e)
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            ret = g.dstdata['ft']
            if self.activation:
                ret = self.activation(ret)
            return ret


class Attention(nn.Module):

    def __init__(self, hidden_dim, attn_drop):
        """语义层次的注意力

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        """
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain)
        nn.init.xavier_normal_(self.attn, gain)

    def forward(self, h):
        """
        :param h: tensor(N, M, d) 顶点基于不同元路径/类型的嵌入，N为顶点数，M为元路径/类型数
        :return: tensor(N, d) 顶点的最终嵌入
        """
        attn = self.attn_drop(self.attn)
        w = torch.tanh(self.fc(h)).mean(dim=0).matmul(attn.t())
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)
        z = (beta * h).sum(dim=1)
        return z


class NetworkSchemaEncoder(nn.Module):

    def __init__(self, hidden_dim, attn_drop, neighbor_sizes):
        """网络结构视图编码器

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        :param neighbor_sizes: List[int] 各邻居类型的采样个数，长度为邻居类型数S
        """
        super().__init__()
        self.gats = nn.ModuleList([HeCoGATConv(hidden_dim, attn_drop, activation=F.elu) for _ in range(len(neighbor_sizes))])
        self.attn = Attention(hidden_dim, attn_drop)
        self.neighbor_sizes = neighbor_sizes

    def forward(self, bgs, feats):
        """
        :param bgs: List[DGLGraph] 各类型邻居到目标顶点的二分图
        :param feats: List[tensor(N_i, d)] 输入顶点特征，feats[0]为目标顶点特征，feats[i]对应bgs[i-1]
        :return: tensor(N_i, d) 目标顶点的最终嵌入
        """
        h = []
        for i in range(len(self.neighbor_sizes)):
            nodes = {bgs[i].dsttypes[0]: bgs[i].dstnodes()}
            sg = sample_neighbors(bgs[i], nodes, self.neighbor_sizes[i])
            h.append(self.gats[i](sg, feats[i + 1], feats[0]))
        h = torch.stack(h, dim=1)
        z_sc = self.attn(h)
        return z_sc


class MetapathEncoder(nn.Module):

    def __init__(self, num_metapaths, hidden_dim, attn_drop):
        """元路径视图编码器

        :param num_metapaths: int 元路径数量M
        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        """
        super().__init__()
        self.gcns = nn.ModuleList([GraphConv(hidden_dim, hidden_dim, norm='right', activation=nn.PReLU()) for _ in range(num_metapaths)])
        self.attn = Attention(hidden_dim, attn_drop)

    def forward(self, mgs, feat):
        """
        :param mgs: List[DGLGraph] 基于元路径的邻居图
        :param feat: tensor(N, d) 输入顶点特征
        :return: tensor(N, d) 输出顶点特征
        """
        h = [gcn(mg, feat) for gcn, mg in zip(self.gcns, mgs)]
        h = torch.stack(h, dim=1)
        z_mp = self.attn(h)
        return z_mp


class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lambda_):
        """对比损失模块

        :param hidden_dim: int 隐含特征维数
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim))
        self.tau = tau
        self.lambda_ = lambda_
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain)

    def sim(self, x, y):
        """计算相似度矩阵

        :param x: tensor(N, d)
        :param y: tensor(N, d)
        :return: tensor(N, N) S[i, j] = exp(cos(x[i], y[j]))
        """
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        numerator = torch.mm(x, y.t())
        denominator = torch.mm(x_norm, y_norm.t())
        return torch.exp(numerator / denominator / self.tau)

    def forward(self, z_sc, z_mp, pos):
        """
        :param z_sc: tensor(N, d) 目标顶点在网络结构视图下的嵌入
        :param z_mp: tensor(N, d) 目标顶点在元路径视图下的嵌入
        :param pos: tensor(N, N) 0-1张量，每个顶点的正样本
        :return: float 对比损失
        """
        z_sc_proj = self.proj(z_sc)
        z_mp_proj = self.proj(z_mp)
        sim_sc2mp = self.sim(z_sc_proj, z_mp_proj)
        sim_mp2sc = sim_sc2mp.t()
        sim_sc2mp = sim_sc2mp / (sim_sc2mp.sum(dim=1, keepdim=True) + 1e-08)
        loss_sc = -torch.log(torch.sum(sim_sc2mp * pos, dim=1)).mean()
        sim_mp2sc = sim_mp2sc / (sim_mp2sc.sum(dim=1, keepdim=True) + 1e-08)
        loss_mp = -torch.log(torch.sum(sim_mp2sc * pos, dim=1)).mean()
        return self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp


class HeCo(nn.Module):

    def __init__(self, in_dims, hidden_dim, feat_drop, attn_drop, neighbor_sizes, num_metapaths, tau, lambda_):
        """HeCo模型

        :param in_dims: List[int] 输入特征维数，in_dims[0]对应目标顶点
        :param hidden_dim: int 隐含特征维数
        :param feat_drop: float 输入特征dropout
        :param attn_drop: float 注意力dropout
        :param neighbor_sizes: List[int] 各邻居类型到采样个数，长度为邻居类型数S
        :param num_metapaths: int 元路径数量M
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for in_dim in in_dims])
        self.feat_drop = nn.Dropout(feat_drop)
        self.sc_encoder = NetworkSchemaEncoder(hidden_dim, attn_drop, neighbor_sizes)
        self.mp_encoder = MetapathEncoder(num_metapaths, hidden_dim, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lambda_)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for fc in self.fcs:
            nn.init.xavier_normal_(fc.weight, gain)

    def forward(self, bgs, mgs, feats, pos):
        """
        :param bgs: List[DGLGraph] 各类型邻居到目标顶点的二分图
        :param mgs: List[DGLGraph] 基于元路径的邻居图
        :param feats: List[tensor(N_i, d_in)] 输入顶点特征，feats[0]为目标顶点特征，feats[i]对应bgs[i-1]
        :param pos: tensor(N_tgt, N_tgt) 布尔张量，每个顶点的正样本
        :return: float 对比损失
        """
        h = [F.elu(self.feat_drop(fc(feat))) for fc, feat in zip(self.fcs, feats)]
        z_sc = self.sc_encoder(bgs, h)
        z_mp = self.mp_encoder(mgs, h[0])
        loss = self.contrast(z_sc, z_mp, pos)
        return loss

    @torch.no_grad()
    def get_embeds(self, mgs, feat):
        """计算目标顶点的最终嵌入(z_mp)

        :param mgs: List[DGLGraph] 基于元路径的邻居图
        :param feat: tensor(N_tgt, d_in) 目标顶点的输入特征
        :return: tensor(N_tgt, d_hid) 目标顶点的最终嵌入
        """
        h = F.elu(self.fcs[0](feat))
        z_mp = self.mp_encoder(mgs, h)
        return z_mp


class ContentAggregation(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        """异构内容嵌入模块，针对一种顶点类型，将该类型顶点的多个输入特征编码为一个向量

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 内容嵌入维数
        """
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, feats):
        """
        :param feats: tensor(N, C, d_in) 输入特征列表，N为batch大小，C为输入特征个数
        :return: tensor(N, d_hid) 顶点的异构内容嵌入向量
        """
        out, _ = self.lstm(feats)
        return torch.mean(out, dim=1)


class NeighborAggregation(nn.Module):

    def __init__(self, emb_dim):
        """邻居聚集模块，针对一种邻居类型t，将一个顶点的所有t类型邻居的内容嵌入向量聚集为一个向量

        :param emb_dim: int 内容嵌入维数
        """
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, emb_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, embeds):
        """
        :param embeds: tensor(N, Nt, d) 邻居的内容嵌入，N为batch大小，Nt为每个顶点的邻居个数
        :return: tensor(N, d) 顶点的t类型邻居聚集嵌入
        """
        out, _ = self.lstm(embeds)
        return torch.mean(out, dim=1)


class TypesCombination(nn.Module):

    def __init__(self, emb_dim):
        """类型组合模块，针对一种顶点类型，将该类型顶点的所有类型的邻居聚集嵌入组合为一个向量

        :param emb_dim: int 邻居嵌入维数
        """
        super().__init__()
        self.attn = nn.Parameter(torch.ones(1, 2 * emb_dim))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, content_embed, neighbor_embeds):
        """
        :param content_embed: tensor(N, d) 内容嵌入，N为batch大小
        :param neighbor_embeds: tensor(A, N, d) 邻居嵌入，A为邻居类型数
        :return: tensor(N, d) 最终嵌入
        """
        neighbor_embeds = torch.cat([content_embed.unsqueeze(0), neighbor_embeds], dim=0)
        cat_embeds = torch.cat([content_embed.repeat(neighbor_embeds.shape[0], 1, 1), neighbor_embeds], dim=-1)
        attn_scores = self.leaky_relu((self.attn * cat_embeds).sum(dim=-1, keepdim=True))
        attn_scores = F.softmax(attn_scores, dim=0)
        out = (attn_scores * neighbor_embeds).sum(dim=0)
        return out


def stack_reducer(nodes):
    return {'nc': nodes.mailbox['m']}


class HetGNN(nn.Module):

    def __init__(self, in_dim, hidden_dim, ntypes):
        """HetGNN模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param ntypes: List[str] 顶点类型列表
        """
        super().__init__()
        self.content_aggs = nn.ModuleDict({ntype: ContentAggregation(in_dim, hidden_dim) for ntype in ntypes})
        self.neighbor_aggs = nn.ModuleDict({ntype: NeighborAggregation(hidden_dim) for ntype in ntypes})
        self.combs = nn.ModuleDict({ntype: TypesCombination(hidden_dim) for ntype in ntypes})

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, C_i, d_in)] 顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_hid)] 顶点类型到输出特征的映射
        """
        with g.local_scope():
            for ntype in g.ntypes:
                g.nodes[ntype].data['c'] = self.content_aggs[ntype](feats[ntype])
            neighbor_embeds = {}
            for dt in g.ntypes:
                tmp = []
                for st in g.ntypes:
                    g.multi_update_all({f'{st}-{dt}': (fn.copy_u('c', 'm'), stack_reducer)}, 'sum')
                    tmp.append(self.neighbor_aggs[st](g.nodes[dt].data.pop('nc')))
                neighbor_embeds[dt] = torch.stack(tmp)
            out = {ntype: self.combs[ntype](g.nodes[ntype].data['c'], neighbor_embeds[ntype]) for ntype in g.ntypes}
            return out

    def calc_score(self, g, h):
        """计算图中每一条边的得分 s(u, v)=h(u)^T h(v)

        :param g: DGLGraph 异构图
        :param h: Dict[str, tensor(N_i, d)] 顶点类型到顶点嵌入的映射
        :return: tensor(A*E) 所有边的得分
        """
        with g.local_scope():
            g.ndata['h'] = h
            for etype in g.etypes:
                g.apply_edges(fn.u_dot_v('h', 'h', 's'), etype=etype)
            return torch.cat(list(g.edata['s'].values())).squeeze(dim=-1)


class MicroConv(nn.Module):

    def __init__(self, out_dim, num_heads, fc_src, fc_dst, attn_src, feat_drop=0.0, negative_slope=0.2, activation=None):
        """微观层次卷积

        针对一种关系（边类型）R=<stype, etype, dtype>，聚集关系R下的邻居信息，得到关系R关于dtype类型顶点的表示
        （特征转换矩阵和注意力向量是与顶点类型相关的，除此之外与GAT完全相同）

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param fc_src: nn.Linear(d_in, K*d_out) 源顶点特征转换模块
        :param fc_dst: nn.Linear(d_in, K*d_out) 目标顶点特征转换模块
        :param attn_src: nn.Parameter(K, 2d_out) 源顶点类型对应的注意力向量
        :param feat_drop: float, optional 输入特征Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param activation: callable, optional 用于输出特征的激活函数，默认为None
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc_src = fc_src
        self.fc_dst = fc_dst
        self.attn_src = attn_src
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

    def forward(self, g, feat):
        """
        :param g: DGLGraph 二分图（只包含一种关系）
        :param feat: tensor(N_src, d_in) or (tensor(N_src, d_in), tensor(N_dst, d_in)) 输入特征
        :return: tensor(N_dst, K*d_out) 该关系关于目标顶点的表示
        """
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, g)
            feat_src = self.fc_src(self.feat_drop(feat_src)).view(-1, self.num_heads, self.out_dim)
            feat_dst = self.fc_dst(self.feat_drop(feat_dst)).view(-1, self.num_heads, self.out_dim)
            el = (feat_src * self.attn_src[:, :self.out_dim]).sum(dim=-1, keepdim=True)
            er = (feat_dst * self.attn_src[:, self.out_dim:]).sum(dim=-1, keepdim=True)
            g.srcdata.update({'ft': feat_src, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = edge_softmax(g, e)
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            ret = g.dstdata['ft'].view(-1, self.num_heads * self.out_dim)
            if self.activation:
                ret = self.activation(ret)
            return ret


class MacroConv(nn.Module):

    def __init__(self, out_dim, num_heads, fc_node, fc_rel, attn, dropout=0.0, negative_slope=0.2):
        """宏观层次卷积

        针对所有关系（边类型），将每种类型的顶点关联的所有关系关于该类型顶点的表示组合起来

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param fc_node: Dict[str, nn.Linear(d_in, K*d_out)] 顶点类型到顶点特征转换模块的映射
        :param fc_rel: Dict[str, nn.Linear(K*d_out, K*d_out)] 关系到关系表示转换模块的映射
        :param attn: nn.Parameter(K, 2d_out)
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc_node = fc_node
        self.fc_rel = fc_rel
        self.attn = attn
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, node_feats, rel_feats):
        """
        :param node_feats: Dict[str, tensor(N_i, d_in) 顶点类型到输入顶点特征的映射
        :param rel_feats: Dict[(str, str, str), tensor(N_i, K*d_out)]
         关系(stype, etype, dtype)到关系关于其终点类型的表示的映射
        :return: Dict[str, tensor(N_i, K*d_out)] 顶点类型到最终顶点嵌入的映射
        """
        node_feats = {ntype: self.fc_node[ntype](feat).view(-1, self.num_heads, self.out_dim) for ntype, feat in node_feats.items()}
        rel_feats = {r: self.fc_rel[r[1]](feat).view(-1, self.num_heads, self.out_dim) for r, feat in rel_feats.items()}
        out_feats = {}
        for ntype, node_feat in node_feats.items():
            rel_node_feats = [feat for rel, feat in rel_feats.items() if rel[2] == ntype]
            if not rel_node_feats:
                continue
            elif len(rel_node_feats) == 1:
                out_feats[ntype] = rel_node_feats[0].view(-1, self.num_heads * self.out_dim)
            else:
                rel_node_feats = torch.stack(rel_node_feats, dim=0)
                cat_feats = torch.cat((node_feat.repeat(rel_node_feats.shape[0], 1, 1, 1), rel_node_feats), dim=-1)
                attn_scores = self.leaky_relu((self.attn * cat_feats).sum(dim=-1, keepdim=True))
                attn_scores = F.softmax(attn_scores, dim=0)
                out_feat = (attn_scores * rel_node_feats).sum(dim=0)
                out_feats[ntype] = self.dropout(out_feat.reshape(-1, self.num_heads * self.out_dim))
        return out_feats


class HGConvLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, ntypes, etypes, dropout=0.0, residual=True):
        """HGConv层

        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param dropout: float, optional Dropout概率，默认为0
        :param residual: bool, optional 是否使用残差连接，默认True
        """
        super().__init__()
        micro_fc = {ntype: nn.Linear(in_dim, num_heads * out_dim, bias=False) for ntype in ntypes}
        micro_attn = {ntype: nn.Parameter(torch.FloatTensor(size=(num_heads, 2 * out_dim))) for ntype in ntypes}
        macro_fc_node = nn.ModuleDict({ntype: nn.Linear(in_dim, num_heads * out_dim, bias=False) for ntype in ntypes})
        macro_fc_rel = nn.ModuleDict({r[1]: nn.Linear(num_heads * out_dim, num_heads * out_dim, bias=False) for r in etypes})
        macro_attn = nn.Parameter(torch.FloatTensor(size=(num_heads, 2 * out_dim)))
        self.micro_conv = nn.ModuleDict({etype: MicroConv(out_dim, num_heads, micro_fc[stype], micro_fc[dtype], micro_attn[stype], dropout, activation=F.relu) for stype, etype, dtype in etypes})
        self.macro_conv = MacroConv(out_dim, num_heads, macro_fc_node, macro_fc_rel, macro_attn, dropout)
        self.residual = residual
        if residual:
            self.res_fc = nn.ModuleDict({ntype: nn.Linear(in_dim, num_heads * out_dim) for ntype in ntypes})
            self.res_weight = nn.ParameterDict({ntype: nn.Parameter(torch.rand(1)) for ntype in ntypes})
        self.reset_parameters(micro_fc, micro_attn, macro_fc_node, macro_fc_rel, macro_attn)

    def reset_parameters(self, micro_fc, micro_attn, macro_fc_node, macro_fc_rel, macro_attn):
        gain = nn.init.calculate_gain('relu')
        for ntype in micro_fc:
            nn.init.xavier_normal_(micro_fc[ntype].weight, gain=gain)
            nn.init.xavier_normal_(micro_attn[ntype], gain=gain)
            nn.init.xavier_normal_(macro_fc_node[ntype].weight, gain=gain)
            if self.residual:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for etype in macro_fc_rel:
            nn.init.xavier_normal_(macro_fc_rel[etype].weight, gain=gain)
        nn.init.xavier_normal_(macro_attn, gain=gain)

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射
        :return: Dict[str, tensor(N_i, K*d_out)] 顶点类型到最终顶点嵌入的映射
        """
        if g.is_block:
            feats_dst = {ntype: feats[ntype][:g.num_dst_nodes(ntype)] for ntype in feats}
        else:
            feats_dst = feats
        rel_feats = {(stype, etype, dtype): self.micro_conv[etype](g[stype, etype, dtype], (feats[stype], feats_dst[dtype])) for stype, etype, dtype in g.canonical_etypes if g.num_edges((stype, etype, dtype)) > 0}
        out_feats = self.macro_conv(feats_dst, rel_feats)
        if self.residual:
            for ntype in out_feats:
                alpha = torch.sigmoid(self.res_weight[ntype])
                inherit_feat = self.res_fc[ntype](feats_dst[ntype])
                out_feats[ntype] = alpha * out_feats[ntype] + (1 - alpha) * inherit_feat
        return out_feats


class HGConv(nn.Module):

    def __init__(self, in_dims, hidden_dim, out_dim, num_heads, ntypes, etypes, predict_ntype, num_layers, dropout=0.0, residual=True):
        """HGConv模型

        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param predict_ntype: str 待预测顶点类型
        :param num_layers: int 层数
        :param dropout: float, optional Dropout概率，默认为0
        :param residual: bool, optional 是否使用残差连接，默认True
        """
        super().__init__()
        self.predict_ntype = predict_ntype
        self.fc_in = nn.ModuleDict({ntype: nn.Linear(in_dim, num_heads * hidden_dim) for ntype, in_dim in in_dims.items()})
        self.layers = nn.ModuleList([HGConvLayer(num_heads * hidden_dim, hidden_dim, num_heads, ntypes, etypes, dropout, residual) for _ in range(num_layers)])
        self.classifier = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in_i)] 顶点类型到输入顶点特征的映射
        :return: tensor(N_i, d_out) 待预测顶点的最终嵌入
        """
        feats = {ntype: self.fc_in[ntype](feat) for ntype, feat in feats.items()}
        for i in range(len(self.layers)):
            feats = self.layers[i](g, feats)
        return self.classifier(feats[self.predict_ntype])


class HGTAttention(nn.Module):

    def __init__(self, out_dim, num_heads, k_linear, q_linear, v_linear, w_att, w_msg, mu):
        """HGT注意力模块

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param k_linear: nn.Linear(d_in, d_out)
        :param q_linear: nn.Linear(d_in, d_out)
        :param v_linear: nn.Linear(d_in, d_out)
        :param w_att: tensor(K, d_out/K, d_out/K)
        :param w_msg: tensor(K, d_out/K, d_out/K)
        :param mu: tensor(1)
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.k_linear = k_linear
        self.q_linear = q_linear
        self.v_linear = v_linear
        self.w_att = w_att
        self.w_msg = w_msg
        self.mu = mu

    def forward(self, g, feat):
        """
        :param g: DGLGraph 二分图（只包含一种关系）
        :param feat: tensor(N_src, d_in) or (tensor(N_src, d_in), tensor(N_dst, d_in)) 输入特征
        :return: tensor(N_dst, d_out) 目标顶点该关于关系的表示
        """
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, g)
            k = self.k_linear(feat_src).view(-1, self.num_heads, self.d_k)
            v = self.v_linear(feat_src).view(-1, self.num_heads, self.d_k)
            q = self.q_linear(feat_dst).view(-1, self.num_heads, self.d_k)
            k = torch.einsum('nhi,hij->nhj', k, self.w_att)
            v = torch.einsum('nhi,hij->nhj', v, self.w_msg)
            g.srcdata.update({'k': k, 'v': v})
            g.dstdata['q'] = q
            g.apply_edges(fn.v_dot_u('q', 'k', 't'))
            attn = g.edata.pop('t').squeeze(dim=-1) * self.mu / math.sqrt(self.d_k)
            attn = edge_softmax(g, attn)
            g.edata['t'] = attn.unsqueeze(dim=-1)
            g.update_all(fn.u_mul_e('v', 't', 'm'), fn.sum('m', 'h'))
            out = g.dstdata['h'].view(-1, self.out_dim)
            return out


class HGTLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, ntypes, etypes, dropout=0.2, use_norm=True):
        """HGT层

        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param dropout: dropout: float, optional Dropout概率，默认为0.2
        :param use_norm: bool, optional 是否使用层归一化，默认为True
        """
        super().__init__()
        d_k = out_dim // num_heads
        k_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}
        q_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}
        v_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}
        w_att = {r[1]: nn.Parameter(torch.Tensor(num_heads, d_k, d_k)) for r in etypes}
        w_msg = {r[1]: nn.Parameter(torch.Tensor(num_heads, d_k, d_k)) for r in etypes}
        mu = {r[1]: nn.Parameter(torch.ones(num_heads)) for r in etypes}
        self.reset_parameters(w_att, w_msg)
        self.conv = HeteroGraphConv({etype: HGTAttention(out_dim, num_heads, k_linear[stype], q_linear[dtype], v_linear[stype], w_att[etype], w_msg[etype], mu[etype]) for stype, etype, dtype in etypes}, 'mean')
        self.a_linear = nn.ModuleDict({ntype: nn.Linear(out_dim, out_dim) for ntype in ntypes})
        self.skip = nn.ParameterDict({ntype: nn.Parameter(torch.ones(1)) for ntype in ntypes})
        self.drop = nn.Dropout(dropout)
        self.use_norm = use_norm
        if use_norm:
            self.norms = nn.ModuleDict({ntype: nn.LayerNorm(out_dim) for ntype in ntypes})

    def reset_parameters(self, w_att, w_msg):
        for etype in w_att:
            nn.init.xavier_uniform_(w_att[etype])
            nn.init.xavier_uniform_(w_msg[etype])

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到输出特征的映射
        """
        if g.is_block:
            feats_dst = {ntype: feats[ntype][:g.num_dst_nodes(ntype)] for ntype in feats}
        else:
            feats_dst = feats
        with g.local_scope():
            hs = self.conv(g, (feats, feats))
            out_feats = {}
            for ntype in g.dsttypes:
                if g.num_dst_nodes(ntype) == 0:
                    continue
                alpha = torch.sigmoid(self.skip[ntype])
                trans_out = self.drop(self.a_linear[ntype](hs[ntype]))
                out = alpha * trans_out + (1 - alpha) * feats_dst[ntype]
                out_feats[ntype] = self.norms[ntype](out) if self.use_norm else out
            return out_feats


class RelativeTemporalEncoding(nn.Module):

    def __init__(self, dim, t_max=240):
        """相对时间编码

        .. math::
          Base(\\Delta T, 2i) = \\sin(\\Delta T / 10000^{2i/d}) \\\\
          Base(\\Delta T, 2i+1) = \\sin(\\Delta T / 10000^{2i+1/d}) \\\\
          RTE(\\Delta T) = T-Linear(Base(\\Delta T))

        :param dim: int 编码维数
        :param t_max: int ΔT∈[0, t_max)
        """
        super().__init__()
        t = torch.arange(t_max).unsqueeze(1)
        denominator = torch.exp(torch.arange(dim) * math.log(10000.0) / dim)
        self.base = t / denominator
        self.base[:, 0::2] = torch.sin(self.base[:, 0::2])
        self.base[:, 1::2] = torch.cos(self.base[:, 1::2])
        self.t_linear = nn.Linear(dim, dim)

    def forward(self, delta_t):
        """返回ΔT对应的相对时间编码"""
        return self.t_linear(self.base[delta_t])


class HGT(nn.Module):

    def __init__(self, in_dims, hidden_dim, out_dim, num_heads, ntypes, etypes, predict_ntype, num_layers, dropout=0.2, use_norm=True):
        """HGT模型

        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param predict_ntype: str 待预测顶点类型
        :param num_layers: int 层数
        :param dropout: dropout: float, optional Dropout概率，默认为0.2
        :param use_norm: bool, optional 是否使用层归一化，默认为True
        """
        super().__init__()
        self.predict_ntype = predict_ntype
        self.adapt_fcs = nn.ModuleDict({ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()})
        self.layers = nn.ModuleList([HGTLayer(hidden_dim, hidden_dim, num_heads, ntypes, etypes, dropout, use_norm) for _ in range(num_layers)])
        self.predict = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射
        :return: tensor(N_i, d_out) 待预测顶点的最终嵌入
        """
        hs = {ntype: F.gelu(self.adapt_fcs[ntype](feats[ntype])) for ntype in feats}
        for layer in self.layers:
            hs = layer(g, hs)
        out = self.predict(hs[self.predict_ntype])
        return out


class MetapathInstanceEncoder(nn.Module):
    """元路径实例编码器，将一个元路径实例所有中间顶点的特征编码为一个向量。"""

    def forward(self, feat):
        """
        :param feat: tensor(E, L, d_in)
        :return: tensor(E, d_out)
        """
        raise NotImplementedError


class MeanEncoder(MetapathInstanceEncoder):

    def __init__(self, in_dim, out_dim):
        super().__init__()

    def forward(self, feat):
        return feat.mean(dim=1)


class LinearEncoder(MetapathInstanceEncoder):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, feat):
        return self.fc(feat.mean(dim=1))


ENCODERS = {'mean': MeanEncoder, 'linear': LinearEncoder}


def get_encoder(name, in_dim, out_dim):
    if name in ENCODERS:
        return ENCODERS[name](in_dim, out_dim)
    else:
        raise ValueError('非法编码器名称{}，可选项为{}'.format(name, list(ENCODERS.keys())))


class IntraMetapathAggregation(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, encoder, attn_drop=0.0, negative_slope=0.01, activation=None):
        """元路径内的聚集

        针对一种顶点类型和 **一个** 首尾为该类型的元路径，将每个目标顶点所有给定元路径的实例编码为一个向量

        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.01
        :param activation: callable, optional 用于输出特征的激活函数，默认为None
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.encoder = get_encoder(encoder, in_dim, num_heads * out_dim)
        self.attn_l = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        self.attn_r = nn.Linear(in_dim, num_heads, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.attn_l, nn.init.calculate_gain('relu'))

    def forward(self, g, node_feat, edge_feat):
        """
        :param g: DGLGraph 基于给定元路径的邻居组成的图，每条边表示一个元路径实例
        :param node_feat: tensor(N, d_in) 输入顶点特征，N为g的终点个数
        :param edge_feat: tensor(E, L, d_in) 元路径实例特征（由中间顶点的特征组成），E为g的边数，L为元路径长度
        :return: tensor(N, K, d_out) 输出顶点特征，K为注意力头数
        """
        with g.local_scope():
            edge_feat = self.encoder(edge_feat)
            edge_feat = edge_feat.view(-1, self.num_heads, self.out_dim)
            el = (edge_feat * self.attn_l).sum(dim=-1).unsqueeze(dim=-1)
            er = self.attn_r(node_feat).unsqueeze(dim=-1)
            g.edata.update({'ft': edge_feat, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.e_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))
            g.update_all(lambda edges: {'m': edges.data['ft'] * edges.data['a']}, fn.sum('m', 'ft'))
            ret = g.dstdata['ft']
            if self.activation:
                ret = self.activation(ret)
            return ret


class InterMetapathAggregation(nn.Module):

    def __init__(self, in_dim, attn_hidden_dim):
        """元路径间的聚集

        针对一种顶点类型和所有首尾为该类型的元路径，将每个顶点关于所有元路径的嵌入组合起来

        :param in_dim: int 顶点关于元路径的嵌入维数（对应元路径内的聚集模块的输出维数）
        :param attn_hidden_dim: int 中间隐含向量维数
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, attn_hidden_dim)
        self.fc2 = nn.Linear(attn_hidden_dim, 1, bias=False)

    def forward(self, z):
        """
        :param z: tensor(N, M, d_in) 每个顶点关于所有元路径的嵌入，N为顶点数，M为元路径个数
        :return: tensor(N, d_in) 聚集后的顶点嵌入
        """
        s = torch.tanh(self.fc1(z)).mean(dim=0)
        e = self.fc2(s)
        beta = e.softmax(dim=0)
        beta = beta.reshape((1, -1, 1))
        z = (beta * z).sum(dim=1)
        return z


class MAGNNLayerNtypeSpecific(nn.Module):

    def __init__(self, num_metapaths, in_dim, out_dim, num_heads, encoder, attn_hidden_dim=128, attn_drop=0.0):
        """特定顶点类型的MAGNN层

        针对一种顶点类型和所有首尾为该类型的元路径，分别对每条元路径做元路径内的聚集，之后做元路径间的聚集

        :param num_metapaths: int 元路径个数
        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param attn_hidden_dim: int, optional 元路径间的聚集中间隐含向量维数，默认为128
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        """
        super().__init__()
        self.intra_metapath_aggs = nn.ModuleList([IntraMetapathAggregation(in_dim, out_dim, num_heads, encoder, attn_drop, activation=F.elu) for _ in range(num_metapaths)])
        self.inter_metapath_agg = InterMetapathAggregation(num_heads * out_dim, attn_hidden_dim)

    def forward(self, gs, node_feat, edge_feat_name):
        """
        :param gs: List[DGLGraph] 基于每条元路径的邻居组成的图
        :param node_feat: tensor(N, d_in) 输入顶点特征，N为给定类型的顶点个数
        :param edge_feat_name: str 元路径实例特征所在的边属性名称
        :return: tensor(N, K*d_out) 最终顶点嵌入，K为注意力头数
        """
        if gs[0].is_block:
            node_feat = node_feat[gs[0].dstdata[dgl.NID]]
        metapath_embeds = [agg(g, node_feat, g.edata[edge_feat_name]).flatten(start_dim=1) for agg, g in zip(self.intra_metapath_aggs, gs)]
        metapath_embeds = torch.stack(metapath_embeds, dim=1)
        return self.inter_metapath_agg(metapath_embeds)


def metapath_instance_feat(metapath, node_feats, instances):
    """返回元路径实例特征，由中间顶点的特征组成。

    :param metapath: List[str] 元路径，顶点类型列表
    :param node_feats: Dict[str, tendor(N_i, d)] 顶点类型到顶点特征的映射，所有类型顶点的特征应具有相同的维数d
    :param instances: tensor(E, L) 元路径实例，E为元路径实例个数，L为元路径长度
    :return: tensor(E, L, d) 元路径实例特征
    """
    feat_dim = node_feats[metapath[0]].shape[1]
    inst_feat = torch.zeros(instances.shape + (feat_dim,))
    for i, ntype in enumerate(metapath):
        inst_feat[:, i] = node_feats[ntype][instances[:, i]]
    return inst_feat


class MAGNNLayer(nn.Module):

    def __init__(self, metapaths, in_dim, out_dim, num_heads, encoder, attn_drop=0.0):
        """MAGNN层

        :param metapaths: Dict[str, List[List[str]]] 顶点类型到其对应的元路径的映射，元路径表示为顶点类型列表
        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        """
        super().__init__()
        self.metapaths = metapaths
        self.layers = nn.ModuleDict({ntype: MAGNNLayerNtypeSpecific(len(metapaths[ntype]), in_dim, out_dim, num_heads, encoder, attn_drop=attn_drop) for ntype in metapaths})
        self.fc = nn.Linear(num_heads * out_dim, out_dim)

    def forward(self, gs, node_feats):
        """
        :param gs: Dict[str, List[DGLGraph]] 顶点类型到其对应的基于每条元路径的邻居组成的图的映射
        :param node_feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射，N_i为对应类型的顶点个数
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到最终顶点嵌入的映射
        """
        self._calc_metapath_instance_feat(gs, node_feats)
        return {ntype: self.fc(self.layers[ntype](gs[ntype], node_feats[ntype], 'feat')) for ntype in gs}

    def _calc_metapath_instance_feat(self, gs, node_feats):
        for ntype in self.metapaths:
            for g, metapath in zip(gs[ntype], self.metapaths[ntype]):
                g.edata['feat'] = metapath_instance_feat(metapath, node_feats, g.edata['inst'])


class MAGNNMinibatch(nn.Module):

    def __init__(self, ntype, metapaths, in_dims, hidden_dim, out_dim, num_heads, encoder, dropout=0.0):
        """使用minibatch训练的MAGNN模型，由特征转换、一个MAGNN层和输出层组成。

        仅针对目标顶点类型及其对应的元路径计算嵌入，其他顶点类型仅使用输入特征

        :param ntype: str 目标顶点类型
        :param metapaths: List[List[str]] 目标顶点类型对应的元路径列表，元路径表示为顶点类型列表
        :param in_dims: Dict[str, int] 所有顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.ntype = ntype
        self.feat_trans = nn.ModuleDict({ntype: nn.Linear(in_dims[ntype], hidden_dim) for ntype in in_dims})
        self.feat_drop = nn.Dropout(dropout)
        self.magnn = MAGNNLayer({self.ntype: metapaths}, hidden_dim, out_dim, num_heads, encoder, dropout)

    def forward(self, blocks, node_feats):
        """
        :param blocks: List[DGLBlock] 目标顶点类型对应的每条元路径的邻居组成的图(block)
        :param node_feats: Dict[str, tensor(N_i, d_in)] 所有顶点类型到输入顶点特征的映射，N_i为对应类型的顶点个数
        :return: tensor(N, d_out) 目标顶点类型的最终嵌入
        """
        hs = {ntype: self.feat_drop(trans(node_feats[ntype])) for ntype, trans in self.feat_trans.items()}
        out = self.magnn({self.ntype: blocks}, hs)[self.ntype]
        return out


class MAGNNMultiLayer(nn.Module):

    def __init__(self, num_layers, metapaths, in_dims, hidden_dim, out_dim, num_heads, encoder, dropout=0.0):
        """多层MAGNN模型，由特征转换、多个MAGNN层和输出层组成。

        :param num_layers: int MAGNN层数
        :param metapaths: Dict[str, List[List[str]]] 顶点类型到元路径列表的映射，元路径表示为顶点类型列表
        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: 注意力头数K
        :param encoder: str 元路径实例编码器名称
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.feat_trans = nn.ModuleDict({ntype: nn.Linear(in_dims[ntype], hidden_dim) for ntype in in_dims})
        self.feat_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MAGNNLayer(metapaths, hidden_dim, hidden_dim, num_heads, encoder, dropout) for _ in range(num_layers - 1)])
        self.layers.append(MAGNNLayer(metapaths, hidden_dim, out_dim, num_heads, encoder, dropout))

    def forward(self, gs, node_feats):
        """
        :param gs: Dict[str, List[DGLGraph]] 顶点类型到其对应的基于每条元路径的邻居组成的图的映射
        :param node_feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入顶点特征的映射，N_i为对应类型的顶点个数
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到最终顶点嵌入的映射
        """
        hs = {ntype: self.feat_drop(trans(node_feats[ntype])) for ntype, trans in self.feat_trans.items()}
        for i in range(len(self.layers) - 1):
            hs = self.layers[i](gs, hs)
            hs = {ntype: F.elu(h) for ntype, h in hs.items()}
        return self.layers[-1](gs, hs)


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.neigh_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, pos, neg):
        """给定中心词、正样本和负样本，返回似然函数的相反数（损失）：

        .. math::
          L=-\\log {\\sigma(v_c \\cdot v_p)}-\\sum_{n \\in neg}{\\log {\\sigma(-v_c \\cdot v_n)}}

        :param center: tensor(N) 中心词
        :param pos: tensor(N) 正样本
        :param neg: tensor(N, M) 负样本
        """
        center_embed = self.center_embed(center)
        pos_embed = self.neigh_embed(pos)
        neg_embed = self.neigh_embed(neg)
        pos_score = torch.sum(center_embed * pos_embed, dim=1)
        pos_score = F.logsigmoid(torch.clamp(pos_score, min=-10, max=10))
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze()
        neg_score = torch.sum(F.logsigmoid(torch.clamp(neg_score, min=-10, max=10)), dim=1)
        return -torch.mean(pos_score + neg_score)


class DistMult(nn.Module):

    def __init__(self, num_rels, feat_dim):
        """知识图谱嵌入模型DistMult

        :param num_rels: int 关系个数
        :param feat_dim: int 嵌入维数
        """
        super().__init__()
        self.w_relations = nn.Parameter(torch.Tensor(num_rels, feat_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_relations, gain=nn.init.calculate_gain('relu'))

    def forward(self, embed, head, rel, tail):
        """
        :param embed: tensor(N, d) 实体嵌入
        :param head: tensor(*) 头实体
        :param rel: tensor(*) 关系
        :param tail: tensor(*) 尾实体
        :return: tensor(*) 三元组得分
        """
        return torch.sum(embed[head] * self.w_relations[rel] * embed[tail], dim=1)


class LinkPrediction(nn.Module):

    def __init__(self, num_nodes, hidden_dim, num_rels, num_layers=2, regularizer='basis', num_bases=None, dropout=0.0):
        """R-GCN连接预测模型
        
        :param num_nodes: int 顶点（实体）数
        :param hidden_dim: int 隐含特征维数
        :param num_rels: int 关系个数
        :param num_layers: int, optional R-GCN层数，默认为2
        :param regularizer: str, 'basis'/'bdd' 权重正则化方法，默认为'basis'
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.embed = nn.Embedding(num_nodes, hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RelGraphConv(hidden_dim, hidden_dim, num_rels, regularizer, num_bases, activation=F.relu if i < num_layers - 1 else None, self_loop=True, low_mem=True, dropout=dropout))
        self.score = DistMult(num_rels, hidden_dim)

    def forward(self, g, etypes):
        """
        :param g: DGLGraph 同构图
        :param etypes: tensor(|E|) 边类型
        :return: tensor(N, d_hid) 顶点嵌入
        """
        h = self.embed.weight
        for layer in self.layers:
            h = layer(g, h, etypes)
        return h

    def calc_score(self, embed, triplets):
        """计算三元组得分

        :param embed: tensor(N, d_hid) 顶点（实体）嵌入
        :param triplets: (tensor(*), tensor(*), tensor(*)) 三元组(head, tail, relation)
        :return: tensor(*) 三元组得分
        """
        head, tail, rel = triplets
        return self.score(embed, head, rel, tail)


class RelGraphConvHetero(nn.Module):

    def __init__(self, in_dim, out_dim, rel_names, num_bases=None, weight=True, self_loop=True, activation=None, dropout=0.0):
        """R-GCN层（用于异构图）

        :param in_dim: 输入特征维数
        :param out_dim: 输出特征维数
        :param rel_names: List[str] 关系名称
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param weight: bool, optional 是否进行线性变换，默认为True
        :param self_loop: 是否包括自环消息，默认为True
        :param activation: callable, optional 激活函数，默认为None
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.rel_names = rel_names
        self.self_loop = self_loop
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = HeteroGraphConv({rel: GraphConv(in_dim, out_dim, norm='right', weight=False, bias=False) for rel in rel_names})
        self.use_weight = weight
        if not num_bases:
            num_bases = len(rel_names)
        self.use_basis = weight and 0 < num_bases < len(rel_names)
        if self.use_weight:
            if self.use_basis:
                self.basis = WeightBasis((in_dim, out_dim), num_bases, len(rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(rel_names), in_dim, out_dim))
                nn.init.xavier_uniform_(self.weight, nn.init.calculate_gain('relu'))
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.loop_weight, nn.init.calculate_gain('relu'))

    def forward(self, g, inputs):
        """
        :param g: DGLGraph 异构图
        :param inputs: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到输出特征的映射
        """
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            kwargs = {rel: {'weight': weight[i]} for i, rel in enumerate(self.rel_names)}
        else:
            kwargs = {}
        hs = self.conv(g, inputs, mod_kwargs=kwargs)
        for ntype in hs:
            if self.self_loop:
                hs[ntype] += torch.matmul(inputs[ntype], self.loop_weight)
            if self.activation:
                hs[ntype] = self.activation(hs[ntype])
            hs[ntype] = self.dropout(hs[ntype])
        return hs


class EntityClassification(nn.Module):

    def __init__(self, num_nodes, hidden_dim, out_dim, rel_names, num_hidden_layers=1, num_bases=None, self_loop=True, dropout=0.0):
        """R-GCN实体分类模型

        :param num_nodes: Dict[str, int] 顶点类型到顶点数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param rel_names: List[str] 关系名称
        :param num_hidden_layers: int, optional R-GCN隐藏层数，默认为1
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param self_loop: bool 是否包括自环消息，默认为True
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.embeds = nn.ModuleDict({ntype: nn.Embedding(num_nodes[ntype], hidden_dim) for ntype in num_nodes})
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(RelGraphConvHetero(hidden_dim, hidden_dim, rel_names, num_bases, False, self_loop, F.relu, dropout))
        self.layers.append(RelGraphConvHetero(hidden_dim, out_dim, rel_names, num_bases, True, self_loop, dropout=dropout))
        self.reset_parameters()

    def reset_parameters(self):
        for k in self.embeds:
            nn.init.xavier_uniform_(self.embeds[k].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        """
        :param g: DGLGraph 异构图
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到顶点嵌入的映射
        """
        h = {k: self.embeds[k].weight for k in self.embeds}
        for layer in self.layers:
            h = layer(g, h)
        return h


class RelationGraphConv(nn.Module):

    def __init__(self, out_dim, num_heads, fc_src, fc_dst, fc_rel, feat_drop=0.0, negative_slope=0.2, activation=None):
        """特定关系的卷积

        针对一种关系（边类型）R=<stype, etype, dtype>，聚集关系R下的邻居信息，得到dtype类型顶点在关系R下的表示，
        注意力向量使用关系R的表示

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param fc_src: nn.Linear(d_in, K*d_out) 源顶点特征转换模块
        :param fc_dst: nn.Linear(d_in, K*d_out) 目标顶点特征转换模块
        :param fc_rel: nn.Linear(d_rel, 2*K*d_out) 关系表示转换模块
        :param feat_drop: float, optional 输入特征Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param activation: callable, optional 用于输出特征的激活函数，默认为None
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc_src = fc_src
        self.fc_dst = fc_dst
        self.fc_rel = fc_rel
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

    def forward(self, g, feat, feat_rel):
        """
        :param g: DGLGraph 二分图（只包含一种关系）
        :param feat: tensor(N_src, d_in) or (tensor(N_src, d_in), tensor(N_dst, d_in)) 输入特征
        :param feat_rel: tensor(d_rel) 关系R的表示
        :return: tensor(N_dst, K*d_out) 目标顶点在关系R下的表示
        """
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, g)
            feat_src = self.fc_src(self.feat_drop(feat_src)).view(-1, self.num_heads, self.out_dim)
            feat_dst = self.fc_dst(self.feat_drop(feat_dst)).view(-1, self.num_heads, self.out_dim)
            attn = self.fc_rel(feat_rel).view(self.num_heads, 2 * self.out_dim)
            el = (feat_src * attn[:, :self.out_dim]).sum(dim=-1, keepdim=True)
            er = (feat_dst * attn[:, self.out_dim:]).sum(dim=-1, keepdim=True)
            g.srcdata.update({'ft': feat_src, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = edge_softmax(g, e)
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            ret = g.dstdata['ft'].view(-1, self.num_heads * self.out_dim)
            if self.activation:
                ret = self.activation(ret)
            return ret


class RelationCrossing(nn.Module):

    def __init__(self, out_dim, num_heads, rel_attn, dropout=0.0, negative_slope=0.2):
        """跨关系消息传递

        针对一种关系R=<stype, etype, dtype>，将dtype类型顶点在不同关系下的表示进行组合

        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param rel_attn: nn.Parameter(K, d) 关系R的注意力向量
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rel_attn = rel_attn
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, feats):
        """
        :param feats: tensor(N_R, N, K*d) dtype类型顶点在不同关系下的表示
        :return: tensor(N, K*d) 跨关系消息传递后dtype类型顶点在关系R下的表示
        """
        num_rel = feats.shape[0]
        if num_rel == 1:
            return feats.squeeze(dim=0)
        feats = feats.view(num_rel, -1, self.num_heads, self.out_dim)
        attn_scores = (self.rel_attn * feats).sum(dim=-1, keepdim=True)
        attn_scores = F.softmax(self.leaky_relu(attn_scores), dim=0)
        out = (attn_scores * feats).sum(dim=0)
        out = self.dropout(out.view(-1, self.num_heads * self.out_dim))
        return out


class RelationFusing(nn.Module):

    def __init__(self, node_hidden_dim, rel_hidden_dim, num_heads, w_node, w_rel, dropout=0.0, negative_slope=0.2):
        """关系混合

        针对一种顶点类型，将该类型顶点在不同关系下的表示进行组合

        :param node_hidden_dim: int 顶点隐含特征维数
        :param rel_hidden_dim: int 关系隐含特征维数
        :param num_heads: int 注意力头数K
        :param w_node: Dict[str, tensor(K, d_node, d_node)] 边类型到顶点关于该关系的特征转换矩阵的映射
        :param w_rel: Dict[str, tensor(K, d_rel, d_node)] 边类型到关系的特征转换矩阵的映射
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        """
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.rel_hidden_dim = rel_hidden_dim
        self.num_heads = num_heads
        self.w_node = nn.ParameterDict(w_node)
        self.w_rel = nn.ParameterDict(w_rel)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, node_feats, rel_feats):
        """
        :param node_feats: Dict[str, tensor(N, K*d_node)] 边类型到顶点在该关系下的表示的映射
        :param rel_feats: Dict[str, tensor(K*d_rel)] 边类型到关系的表示的映射
        :return: tensor(N, K*d_node) 该类型顶点的最终嵌入
        """
        etypes = list(node_feats.keys())
        num_rel = len(node_feats)
        if num_rel == 1:
            return node_feats[etypes[0]]
        node_feats = torch.stack([node_feats[e] for e in etypes], dim=0).reshape(num_rel, -1, self.num_heads, self.node_hidden_dim)
        rel_feats = torch.stack([rel_feats[e] for e in etypes], dim=0).reshape(num_rel, self.num_heads, self.rel_hidden_dim)
        w_node = torch.stack([self.w_node[e] for e in etypes], dim=0)
        w_rel = torch.stack([self.w_rel[e] for e in etypes], dim=0)
        node_feats = torch.einsum('rnhk,rhki->rnhi', node_feats, w_node)
        rel_feats = torch.einsum('rhk,rhki->rhi', rel_feats, w_rel)
        attn_scores = (node_feats * rel_feats.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
        attn_scores = F.softmax(self.leaky_relu(attn_scores), dim=0)
        out = (attn_scores * node_feats).sum(dim=0)
        out = self.dropout(out.view(-1, self.num_heads * self.node_hidden_dim))
        return out


class RHGNNLayer(nn.Module):

    def __init__(self, node_in_dim, node_out_dim, rel_in_dim, rel_out_dim, num_heads, ntypes, etypes, dropout=0.0, negative_slope=0.2, residual=True):
        """R-HGNN层

        :param node_in_dim: int 顶点输入特征维数
        :param node_out_dim: int 顶点输出特征维数
        :param rel_in_dim: int 关系输入特征维数
        :param rel_out_dim: int 关系输出特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param residual: bool, optional 是否使用残差连接，默认True
        """
        super().__init__()
        fc_node = {ntype: nn.Linear(node_in_dim, num_heads * node_out_dim, bias=False) for ntype in ntypes}
        fc_rel = {etype: nn.Linear(rel_in_dim, 2 * num_heads * node_out_dim, bias=False) for _, etype, _ in etypes}
        self.rel_graph_conv = nn.ModuleDict({etype: RelationGraphConv(node_out_dim, num_heads, fc_node[stype], fc_node[dtype], fc_rel[etype], dropout, negative_slope, F.relu) for stype, etype, dtype in etypes})
        self.residual = residual
        if residual:
            self.fc_res = nn.ModuleDict({ntype: nn.Linear(node_in_dim, num_heads * node_out_dim) for ntype in ntypes})
            self.res_weight = nn.ParameterDict({ntype: nn.Parameter(torch.rand(1)) for ntype in ntypes})
        self.fc_upd = nn.ModuleDict({etype: nn.Linear(rel_in_dim, num_heads * rel_out_dim) for _, etype, _ in etypes})
        rel_attn = {etype: nn.Parameter(torch.FloatTensor(num_heads, node_out_dim)) for _, etype, _ in etypes}
        self.rel_cross = nn.ModuleDict({etype: RelationCrossing(node_out_dim, num_heads, rel_attn[etype], dropout, negative_slope) for _, etype, _ in etypes})
        self.rev_etype = {e: next(re for rs, re, rd in etypes if rs == d and rd == s) for s, e, d in etypes}
        self.reset_parameters(rel_attn)

    def reset_parameters(self, rel_attn):
        gain = nn.init.calculate_gain('relu')
        for etype in rel_attn:
            nn.init.xavier_normal_(rel_attn[etype], gain=gain)

    def forward(self, g, feats, rel_feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[(str, str, str), tensor(N_i, d_in)] 关系（三元组）到目标顶点输入特征的映射
        :param rel_feats: Dict[str, tensor(d_in_rel)] 边类型到输入关系特征的映射
        :return: Dict[(str, str, str), tensor(N_i, K*d_out)], Dict[str, tensor(K*d_out_rel)]
         关系（三元组）到目标顶点在该关系下的表示的映射、边类型到关系表示的映射
        """
        if g.is_block:
            feats_dst = {r: feats[r][:g.num_dst_nodes(r[2])] for r in feats}
        else:
            feats_dst = feats
        node_rel_feats = {(stype, etype, dtype): self.rel_graph_conv[etype](g[stype, etype, dtype], (feats[dtype, self.rev_etype[etype], stype], feats_dst[stype, etype, dtype]), rel_feats[etype]) for stype, etype, dtype in g.canonical_etypes if g.num_edges((stype, etype, dtype)) > 0}
        if self.residual:
            for stype, etype, dtype in node_rel_feats:
                alpha = torch.sigmoid(self.res_weight[dtype])
                inherit_feat = self.fc_res[dtype](feats_dst[stype, etype, dtype])
                node_rel_feats[stype, etype, dtype] = alpha * node_rel_feats[stype, etype, dtype] + (1 - alpha) * inherit_feat
        out_feats = {}
        for stype, etype, dtype in node_rel_feats:
            dst_node_rel_feats = torch.stack([node_rel_feats[r] for r in node_rel_feats if r[2] == dtype], dim=0)
            out_feats[stype, etype, dtype] = self.rel_cross[etype](dst_node_rel_feats)
        rel_feats = {etype: self.fc_upd[etype](rel_feats[etype]) for etype in rel_feats}
        return out_feats, rel_feats


class RHGNN(nn.Module):

    def __init__(self, in_dims, hidden_dim, out_dim, rel_in_dim, rel_hidden_dim, num_heads, ntypes, etypes, predict_ntype, num_layers, dropout=0.0, negative_slope=0.2, residual=True):
        """R-HGNN模型

        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 顶点隐含特征维数
        :param out_dim: int 顶点输出特征维数
        :param rel_in_dim: int 关系输入特征维数
        :param rel_hidden_dim: int 关系隐含特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param predict_ntype: str 待预测顶点类型
        :param num_layers: int 层数
        :param dropout: float, optional Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param residual: bool, optional 是否使用残差连接，默认True
        """
        super().__init__()
        self._d = num_heads * hidden_dim
        self.etypes = etypes
        self.predict_ntype = predict_ntype
        self.fc_in = nn.ModuleDict({ntype: nn.Linear(in_dim, num_heads * hidden_dim) for ntype, in_dim in in_dims.items()})
        self.rel_embed = nn.ParameterDict({etype: nn.Parameter(torch.FloatTensor(1, rel_in_dim)) for _, etype, _ in etypes})
        self.layers = nn.ModuleList()
        self.layers.append(RHGNNLayer(num_heads * hidden_dim, hidden_dim, rel_in_dim, rel_hidden_dim, num_heads, ntypes, etypes, dropout, negative_slope, residual))
        for _ in range(1, num_layers):
            self.layers.append(RHGNNLayer(num_heads * hidden_dim, hidden_dim, num_heads * rel_hidden_dim, rel_hidden_dim, num_heads, ntypes, etypes, dropout, negative_slope, residual))
        w_node = {etype: nn.Parameter(torch.FloatTensor(num_heads, hidden_dim, hidden_dim)) for _, etype, _ in etypes}
        w_rel = {etype: nn.Parameter(torch.FloatTensor(num_heads, rel_hidden_dim, hidden_dim)) for _, etype, _ in etypes}
        self.rel_fusing = nn.ModuleDict({ntype: RelationFusing(hidden_dim, rel_hidden_dim, num_heads, {e: w_node[e] for _, e, d in etypes if d == ntype}, {e: w_rel[e] for _, e, d in etypes if d == ntype}, dropout, negative_slope) for ntype in ntypes})
        self.classifier = nn.Linear(num_heads * hidden_dim, out_dim)
        self.reset_parameters(self.rel_embed, w_node, w_rel)

    def reset_parameters(self, rel_embed, w_node, w_rel):
        gain = nn.init.calculate_gain('relu')
        for etype in rel_embed:
            nn.init.xavier_normal_(rel_embed[etype], gain=gain)
            nn.init.xavier_normal_(w_node[etype], gain=gain)
            nn.init.xavier_normal_(w_rel[etype], gain=gain)

    def forward(self, blocks, feats):
        """
        :param blocks: blocks: List[DGLBlock]
        :param feats: Dict[str, tensor(N_i, d_in_i)] 顶点类型到输入顶点特征的映射
        :return: tensor(N_i, d_out) 待预测顶点的最终嵌入
        """
        feats = {(stype, etype, dtype): self.fc_in[dtype](feats[dtype]) for stype, etype, dtype in self.etypes}
        rel_feats = {rel: emb.flatten() for rel, emb in self.rel_embed.items()}
        for block, layer in zip(blocks, self.layers):
            feats, rel_feats = layer(block, feats, rel_feats)
        out_feats = {ntype: self.rel_fusing[ntype]({e: feats[s, e, d] for s, e, d in feats if d == ntype}, {e: rel_feats[e] for s, e, d in feats if d == ntype}) for ntype in set(d for _, _, d in feats)}
        return self.classifier(out_feats[self.predict_ntype])

    @torch.no_grad()
    def inference(self, g, feats, device, batch_size):
        """离线推断所有顶点的最终嵌入（不使用邻居采样）

        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in_i)] 顶点类型到输入顶点特征的映射
        :param device: torch.device
        :param batch_size: int 批大小
        :return: tensor(N_i, d_out) 待预测顶点的最终嵌入
        """
        feats = {(stype, etype, dtype): self.fc_in[dtype](feats[dtype]) for stype, etype, dtype in self.etypes}
        rel_feats = {rel: emb.flatten() for rel, emb in self.rel_embed.items()}
        for layer in self.layers:
            embeds = {(stype, etype, dtype): torch.zeros(g.num_nodes(dtype), self._d) for stype, etype, dtype in g.canonical_etypes}
            sampler = MultiLayerFullNeighborSampler(1)
            loader = NodeDataLoader(g, {ntype: torch.arange(g.num_nodes(ntype)) for ntype in g.ntypes}, sampler, batch_size=batch_size, shuffle=True)
            for input_nodes, output_nodes, blocks in tqdm(loader):
                block = blocks[0]
                in_feats = {(s, e, d): feats[s, e, d][input_nodes[d]] for s, e, d in feats}
                h, rel_embeds = layer(block, in_feats, rel_feats)
                for s, e, d in h:
                    embeds[s, e, d][output_nodes[d]] = h[s, e, d].cpu()
            feats = embeds
            rel_feats = rel_embeds
        feats = {r: feat for r, feat in feats.items()}
        out_feats = {ntype: torch.zeros(g.num_nodes(ntype), self._d) for ntype in g.ntypes}
        for ntype in set(d for _, _, d in feats):
            dst_feats = {e: feats[s, e, d] for s, e, d in feats if d == ntype}
            dst_rel_feats = {e: rel_feats[e] for s, e, d in feats if d == ntype}
            for batch in DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=batch_size):
                out_feats[ntype][batch] = self.rel_fusing[ntype]({e: dst_feats[e][batch] for e in dst_rel_feats}, dst_rel_feats)
        return self.classifier(out_feats[self.predict_ntype])


class FeedForwardNet(nn.Module):
    """L层全连接网络"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.0):
        super().__init__()
        self.fc = nn.ModuleList()
        if num_layers == 1:
            self.fc.append(nn.Linear(in_dim, out_dim, bias=False))
        else:
            self.fc.append(nn.Linear(in_dim, hidden_dim, bias=False))
            for _ in range(num_layers - 2):
                self.fc.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.fc.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: tensor(N, d_in)
        :return: tensor(N, d_out)
        """
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i < len(self.fc) - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_hops, num_layers, dropout=0.0):
        """SIGN模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_hops: int 跳数r
        :param num_layers: int 全连接网络层数
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.inception_ffs = nn.ModuleList([FeedForwardNet(in_dim, hidden_dim, hidden_dim, num_layers, dropout) for _ in range(num_hops + 1)])
        self.project = FeedForwardNet((num_hops + 1) * hidden_dim, hidden_dim, out_dim, num_layers, dropout)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats):
        """
        :param feats: List[tensor(N, d_in)] 每一跳的邻居聚集特征，长度为r+1
        :return: tensor(N, d_out) 输出顶点特征
        """
        h = torch.cat([ff(feat) for ff, feat in zip(self.inception_ffs, feats)], dim=-1)
        out = self.project(self.dropout(self.prelu(h)))
        return out


class GraphAttention(nn.Module):
    """图注意力模块，用于计算顶点邻居的重要性"""

    def __init__(self, out_dim, num_heads):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

    def forward(self, g, feat_src, feat_dst):
        """
        :param g: DGLGraph 同构图
        :param feat_src: tensor(N_src, K, d_out) 起点特征
        :param feat_dst: tensor(N_dst, K, d_out) 终点特征
        :return: tensor(E, K, 1) 所有顶点对的邻居重要性
        """
        raise NotImplementedError


class GATOriginalAttention(GraphAttention):
    """原始GAT注意力"""

    def __init__(self, out_dim, num_heads):
        super().__init__(out_dim, num_heads)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, g, feat_src, feat_dst):
        el = (feat_src * self.attn_l).sum(dim=-1, keepdim=True)
        er = (feat_dst * self.attn_r).sum(dim=-1, keepdim=True)
        g.srcdata['el'] = el
        g.dstdata['er'] = er
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        return g.edata.pop('e')


class DotProductAttention(GraphAttention):
    """点积注意力"""

    def forward(self, g, feat_src, feat_dst):
        g.srcdata['ft'] = feat_src
        g.dstdata['ft'] = feat_dst
        g.apply_edges(lambda edges: {'e': torch.sum(edges.src['ft'] * edges.dst['ft'], dim=-1, keepdim=True)})
        return g.edata.pop('e')


class ScaledDotProductAttention(DotProductAttention):

    def forward(self, g, feat_src, feat_dst):
        return super().forward(g, feat_src, feat_dst) / math.sqrt(self.out_dim)


class MixedGraphAttention(GATOriginalAttention, DotProductAttention):

    def forward(self, g, feat_src, feat_dst):
        return GATOriginalAttention.forward(self, g, feat_src, feat_dst) * torch.sigmoid(DotProductAttention.forward(self, g, feat_src, feat_dst))


GRAPH_ATTENTIONS = {'GO': GATOriginalAttention, 'DP': DotProductAttention, 'SD': ScaledDotProductAttention, 'MX': MixedGraphAttention}


def get_graph_attention(attn_type, out_dim, num_heads):
    if attn_type in GRAPH_ATTENTIONS:
        return GRAPH_ATTENTIONS[attn_type](out_dim, num_heads)
    else:
        raise ValueError('非法图注意力类型{}，可选项为{}'.format(attn_type, list(GRAPH_ATTENTIONS.keys())))


class SuperGATConv(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_type, neg_sample_ratio=0.5, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, activation=None):
        """SuperGAT层，自监督任务是连接预测

        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param attn_type: str 注意力类型，可选择GO, DP, SD和MX
        :param neg_sample_ratio: float, optional 负样本边数量占正样本边数量的比例，默认0.5
        :param feat_drop: float, optional 输入特征Dropout概率，默认为0
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        :param activation: callable, optional 用于输出特征的激活函数，默认为None
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.attn = get_graph_attention(attn_type, out_dim, num_heads)
        self.neg_sampler = RatioNegativeSampler(neg_sample_ratio)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

    def forward(self, g, feat):
        """
        :param g: DGLGraph 同构图
        :param feat: tensor(N_src, d_in) 输入顶点特征
        :return: tensor(N_dst, K, d_out) 输出顶点特征
        """
        with g.local_scope():
            feat_src = self.fc(self.feat_drop(feat)).view(-1, self.num_heads, self.out_dim)
            feat_dst = feat_src[:g.num_dst_nodes()] if g.is_block else feat_src
            e = self.leaky_relu(self.attn(g, feat_src, feat_dst))
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))
            g.srcdata['ft'] = feat_src
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            out = g.dstdata['ft']
            if self.training:
                neg_g = dgl.graph(self.neg_sampler(g, list(range(g.num_edges()))), num_nodes=g.num_nodes(), device=g.device)
                neg_e = self.attn(neg_g, feat_src, feat_src)
                self.attn_x = torch.cat([e, neg_e]).squeeze(dim=-1).mean(dim=1)
                self.attn_y = torch.cat([torch.ones(e.shape[0]), torch.zeros(neg_e.shape[0])])
            if self.activation:
                out = self.activation(out)
            return out

    def get_attn_loss(self):
        """返回自监督注意力损失（即连接预测损失）"""
        if self.training:
            return F.binary_cross_entropy_with_logits(self.attn_x, self.attn_y)
        else:
            return torch.tensor(0.0)


class SuperGAT(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, attn_type, neg_sample_ratio=0.5, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2):
        """两层SuperGAT模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param attn_type: str 注意力类型，可选择GO, DP, SD和MX
        :param neg_sample_ratio: float, optional 负样本边数量占正样本边数量的比例，默认0.5
        :param feat_drop: float, optional 输入特征Dropout概率，默认为0
        :param attn_drop: float, optional 注意力权重Dropout概率，默认为0
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.2
        """
        super().__init__()
        self.conv1 = SuperGATConv(in_dim, hidden_dim, num_heads, attn_type, neg_sample_ratio, feat_drop, attn_drop, negative_slope, F.elu)
        self.conv2 = SuperGATConv(num_heads * hidden_dim, out_dim, num_heads, attn_type, neg_sample_ratio, 0, attn_drop, negative_slope)

    def forward(self, g, feat):
        """
        :param g: DGLGraph 同构图
        :param feat: tensor(N, d_in) 输入顶点特征
        :return: tensor(N, d_out), tensor(1) 输出顶点特征和自监督注意力损失
        """
        h = self.conv1(g, feat).flatten(start_dim=1)
        h = self.conv2(g, h).mean(dim=1)
        return h, self.conv1.get_attn_loss() + self.conv2.get_attn_loss()


class KGCNLayer(nn.Module):

    def __init__(self, hidden_dim, aggregator):
        """KGCN层

        :param hidden_dim: int 隐含特征维数d
        :param aggregator: str 实体表示与邻居表示的组合方式：sum, concat, neighbor
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregator = aggregator
        if aggregator == 'concat':
            self.w = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.w = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src_feat, dst_feat, rel_feat, user_feat, activation):
        """
        :param src_feat: tensor(B, K^h, K, d) 输入实体表示，B为batch大小，K为邻居个数，h为跳步数/层数
        :param dst_feat: tensor(B, K^h, d) 目标实体表示
        :param rel_feat: tensor(B, K^h, K, d) 关系表示
        :param user_feat: tensor(B, d) 用户表示
        :param activation: callable 激活函数
        :return: tensor(B, K^h, d) 输出实体表示
        """
        batch_size = user_feat.shape[0]
        user_feat = user_feat.view(batch_size, 1, 1, self.hidden_dim)
        user_rel_scores = (user_feat * rel_feat).sum(dim=-1)
        user_rel_scores = F.softmax(user_rel_scores, dim=-1).unsqueeze(dim=-1)
        agg = (user_rel_scores * src_feat).sum(dim=2)
        if self.aggregator == 'sum':
            out = (dst_feat + agg).view(-1, self.hidden_dim)
        elif self.aggregator == 'concat':
            out = torch.cat([dst_feat, agg], dim=-1).view(-1, 2 * self.hidden_dim)
        else:
            out = agg.view(-1, self.hidden_dim)
        out = self.w(out).view(batch_size, -1, self.hidden_dim)
        return activation(out)


class KGCN(nn.Module):

    def __init__(self, hidden_dim, neighbor_size, aggregator, num_hops, num_users, num_entities, num_rels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.neighbor_size = neighbor_size
        self.num_hops = num_hops
        self.aggregator = KGCNLayer(hidden_dim, aggregator)
        self.user_embed = nn.Embedding(num_users, hidden_dim)
        self.entity_embed = nn.Embedding(num_entities, hidden_dim)
        self.rel_embed = nn.Embedding(num_rels, hidden_dim)

    def forward(self, pair_graph, blocks):
        """
        :param pair_graph: DGLGraph 用户-物品子图
        :param blocks: List[DGLBlock] 知识图谱的MFG，blocks[-1].dstnodes()对应items
        :return: tensor(B) 用户-物品预测概率
        """
        u, v = pair_graph.edges()
        users = pair_graph.nodes['user'].data[dgl.NID][u]
        user_feat = self.user_embed(users)
        entities, relations = self._get_neighbors(v, blocks)
        item_feat = self._aggregate(entities, relations, user_feat)
        scores = (user_feat * item_feat).sum(dim=-1)
        return torch.sigmoid(scores)

    def _get_neighbors(self, v, blocks):
        batch_size = v.shape[0]
        entities, relations = [blocks[-1].dstdata[dgl.NID][v].unsqueeze(dim=-1)], []
        for b in reversed(blocks):
            u, dst = b.in_edges(v)
            entities.append(b.srcdata[dgl.NID][u].view(batch_size, -1))
            relations.append(b.edata['relation'][b.edge_ids(u, dst)].view(batch_size, -1))
            v = u
        return entities, relations

    def _aggregate(self, entities, relations, user_feat):
        batch_size = user_feat.shape[0]
        entity_feats = [self.entity_embed(entity) for entity in entities]
        rel_feats = [self.rel_embed(rel) for rel in relations]
        for h in range(self.num_hops):
            activation = torch.tanh if h == self.num_hops - 1 else torch.sigmoid
            new_entity_feats = [self.aggregator(entity_feats[i + 1].view(batch_size, -1, self.neighbor_size, self.hidden_dim), entity_feats[i], rel_feats[i].view(batch_size, -1, self.neighbor_size, self.hidden_dim), user_feat, activation) for i in range(self.num_hops - h)]
            entity_feats = new_entity_feats
        return entity_feats[0].view(batch_size, self.hidden_dim)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'hidden_dim': 4, 'attn_drop': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContentAggregation,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Contrast,
     lambda: ([], {'hidden_dim': 4, 'tau': 4, 'lambda_': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DistMult,
     lambda: ([], {'num_rels': 4, 'feat_dim': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (FeedForwardNet,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeedforwardNeuralNetwork,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KGCNLayer,
     lambda: ([], {'hidden_dim': 4, 'aggregator': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 1, 1, 4]), _mock_layer()], {}),
     False),
    (LSTM,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'num_layers': 1, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LinearEncoder,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearRegressionModel,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogisticRegressionModel,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MarginLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 64]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MeanEncoder,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NeighborAggregation,
     lambda: ([], {'emb_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (RNN,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'num_layers': 1, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (RelationCrossing,
     lambda: ([], {'out_dim': 4, 'num_heads': 4, 'rel_attn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RelativeTemporalEncoding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {}),
     True),
    (SemanticAttention,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TwoLayerNet,
     lambda: ([], {'D_in': 4, 'H': 4, 'D_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ZZy979_pytorch_tutorial(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

