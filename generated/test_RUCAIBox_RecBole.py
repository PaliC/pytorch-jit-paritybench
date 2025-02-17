import sys
_module = sys.modules[__name__]
del sys
conf = _module
recbole = _module
config = _module
configurator = _module
data = _module
dataloader = _module
abstract_dataloader = _module
general_dataloader = _module
knowledge_dataloader = _module
user_dataloader = _module
dataset = _module
customized_dataset = _module
dataset = _module
decisiontree_dataset = _module
kg_dataset = _module
kg_seq_dataset = _module
sequential_dataset = _module
interaction = _module
transform = _module
utils = _module
evaluator = _module
base_metric = _module
collector = _module
metrics = _module
register = _module
utils = _module
model = _module
abstract_recommender = _module
context_aware_recommender = _module
afm = _module
autoint = _module
dcn = _module
dcnv2 = _module
deepfm = _module
dssm = _module
eulernet = _module
ffm = _module
fignn = _module
fm = _module
fnn = _module
fwfm = _module
kd_dagfm = _module
lr = _module
nfm = _module
pnn = _module
widedeep = _module
xdeepfm = _module
exlib_recommender = _module
lightgbm = _module
xgboost = _module
general_recommender = _module
admmslim = _module
bpr = _module
cdae = _module
convncf = _module
dgcf = _module
diffrec = _module
dmf = _module
ease = _module
enmf = _module
fism = _module
gcmc = _module
itemknn = _module
ldiffrec = _module
lightgcn = _module
line = _module
macridvae = _module
multidae = _module
multivae = _module
nais = _module
nceplrec = _module
ncl = _module
neumf = _module
ngcf = _module
nncf = _module
pop = _module
ract = _module
random = _module
recvae = _module
sgl = _module
simplex = _module
slimelastic = _module
spectralcf = _module
init = _module
knowledge_aware_recommender = _module
cfkg = _module
cke = _module
kgat = _module
kgcn = _module
kgin = _module
kgnnls = _module
ktup = _module
mcclk = _module
mkr = _module
ripplenet = _module
layers = _module
loss = _module
sequential_recommender = _module
bert4rec = _module
caser = _module
core = _module
dien = _module
din = _module
fdsa = _module
fearec = _module
fossil = _module
fpmc = _module
gcsan = _module
gru4rec = _module
gru4reccpr = _module
gru4recf = _module
gru4reckg = _module
hgn = _module
hrm = _module
ksr = _module
lightsans = _module
narm = _module
nextitnet = _module
npe = _module
repeatnet = _module
s3rec = _module
sasrec = _module
sasreccpr = _module
sasrecf = _module
shan = _module
sine = _module
srgnn = _module
stamp = _module
transrec = _module
quick_start = _module
quick_start = _module
sampler = _module
sampler = _module
trainer = _module
hyper_tuning = _module
trainer = _module
argument_list = _module
case_study = _module
enum_type = _module
logger = _module
url = _module
utils = _module
wandblogger = _module
case_study_example = _module
save_and_load_example = _module
session_based_rec_example = _module
run_hyper = _module
run_recbole = _module
run_recbole_group = _module
setup = _module
significance_test = _module
test_command_line = _module
test_config = _module
test_overall = _module
test_dataloader = _module
test_dataset = _module
test_transform = _module
test_evaluation_setting = _module
test_hyper_tuning = _module
test_loss_metrics = _module
test_rank_metrics = _module
test_topk_metrics = _module
test_model_auto = _module
test_model_manual = _module

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


import re


from logging import getLogger


from typing import Literal


import math


import copy


import torch


import numpy as np


from collections import Counter


from collections import defaultdict


import pandas as pd


import torch.nn.utils.rnn as rnn_utils


from scipy.sparse import coo_matrix


import random


from copy import deepcopy


import warnings


import itertools


import torch.nn as nn


from torch.nn.init import xavier_normal_


from torch.nn.init import constant_


import torch.nn.functional as F


from torch.nn.init import xavier_uniform_


from itertools import product


from torch import nn


import scipy.sparse as sp


import random as rd


from torch.autograd import Variable


import enum


import typing


from torch.nn.init import normal_


from sklearn.utils.extmath import randomized_svd


from sklearn.linear_model import ElasticNet


from sklearn.exceptions import ConvergenceWarning


import collections


import torch.nn.functional as fn


from torch.nn import functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import PackedSequence


from torch.nn import Parameter


from torch.nn.init import uniform_


import logging


import torch.distributed as dist


from collections.abc import MutableMapping


from numpy.random import sample


from time import time


import torch.optim as optim


from torch.nn.utils.clip_grad import clip_grad_norm_


import torch.cuda.amp as amp


from torch.nn.parallel import DistributedDataParallel


from torch.utils.tensorboard import SummaryWriter


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\x1b['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\x1b[0m'


class AbstractRecommender(nn.Module):
    """Base class for all models"""

    def __init__(self):
        self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        """Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        """Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        """full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, 'other_parameter_name'):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color('\nTrainable parameters', 'blue') + f': {params}'


class FLEmbedding(nn.Module):
    """Embedding for float fields.

    Args:
        field_dims: list, the number of float in each float fields
        offsets: list, the dimension offset of each float field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,2)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FLEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        base, index = torch.split(input_x, [1, 1], dim=-1)
        index = index.squeeze(-1).long()
        index = index + index.new_tensor(self.offsets).unsqueeze(0)
        output = base * self.embedding(index)
        return output


class FMEmbedding(nn.Module):
    """Embedding for token fields.

    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FMEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class FMFirstOrderLinear(nn.Module):
    """Calculate the first order score of the input features.
    This class is a member of ContextRecommender, you can call it easily when inherit ContextRecommender.

    """

    def __init__(self, config, dataset, output_dim=1):
        super(FMFirstOrderLinear, self).__init__()
        self.field_names = dataset.fields(source=[FeatureSource.INTERACTION, FeatureSource.USER, FeatureSource.USER_ID, FeatureSource.ITEM, FeatureSource.ITEM_ID])
        self.LABEL = config['LABEL_FIELD']
        self.device = config['device']
        self.numerical_features = config['numerical_features']
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.float_seq_field_names = []
        self.float_seq_field_dims = []
        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.FLOAT and field_name in self.numerical_features:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.FLOAT_SEQ and field_name in self.numerical_features:
                self.float_seq_field_names.append(field_name)
                self.float_seq_field_dims.append(dataset.num(field_name))
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(self.token_field_dims, self.token_field_offsets, output_dim)
        if len(self.float_field_dims) > 0:
            self.float_field_offsets = np.array((0, *np.cumsum(self.float_field_dims)[:-1]), dtype=np.long)
            self.float_embedding_table = FLEmbedding(self.float_field_dims, self.float_field_offsets, output_dim)
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(nn.Embedding(token_seq_field_dim, output_dim))
        if len(self.float_seq_field_dims) > 0:
            self.float_seq_embedding_table = nn.ModuleList()
            for float_seq_field_dim in self.float_seq_field_dims:
                self.float_seq_embedding_table.append(nn.Embedding(float_seq_field_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros((output_dim,)), requires_grad=True)

    def embed_float_fields(self, float_fields):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field, 2]
            embed (bool): Return the embedding of columns or just the columns itself. Defaults to ``True``.

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        if float_fields is None:
            return None
        float_embedding = self.float_embedding_table(float_fields)
        float_embedding = torch.sum(float_embedding, dim=1, keepdim=True)
        return float_embedding

    def embed_float_seq_fields(self, float_seq_fields, mode='mean'):
        """Embed the float sequence feature columns

        Args:
            float_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len, 2]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of float sequence columns.
        """
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)
            float_seq_embedding = base * embedding_table(index.long())
            mask = mask.unsqueeze(2).expand_as(float_seq_embedding)
            if mode == 'max':
                masked_float_seq_embedding = float_seq_embedding - (1 - mask) * 1000000000.0
                result = torch.max(masked_float_seq_embedding, dim=1, keepdim=True)
            elif mode == 'sum':
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=1, keepdim=True)
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=1)
                eps = torch.FloatTensor([1e-08])
                result = torch.div(result, value_cnt + eps)
                result = result.unsqueeze(1)
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.sum(torch.cat(fields_result, dim=1), dim=1, keepdim=True)

    def embed_token_fields(self, token_fields):
        """Calculate the first order score of token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The first order score of token feature columns
        """
        if token_fields is None:
            return None
        token_embedding = self.token_embedding_table(token_fields)
        token_embedding = torch.sum(token_embedding, dim=1, keepdim=True)
        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields):
        """Calculate the first order score of token sequence feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]

        Returns:
            torch.FloatTensor: The first order score of token sequence feature columns
        """
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)
            token_seq_embedding = embedding_table(token_seq_field)
            mask = mask.unsqueeze(2).expand_as(token_seq_embedding)
            masked_token_seq_embedding = token_seq_embedding * mask.float()
            result = torch.sum(masked_token_seq_embedding, dim=1, keepdim=True)
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.sum(torch.cat(fields_result, dim=1), dim=1, keepdim=True)

    def forward(self, interaction):
        total_fields_embedding = []
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 3:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)
        else:
            float_fields = None
        float_fields_embedding = self.embed_float_fields(float_fields)
        if float_fields_embedding is not None:
            total_fields_embedding.append(float_fields_embedding)
        float_seq_fields = []
        for field_name in self.float_seq_field_names:
            float_seq_fields.append(interaction[field_name])
        float_seq_fields_embedding = self.embed_float_seq_fields(float_seq_fields)
        if float_seq_fields_embedding is not None:
            total_fields_embedding.append(float_seq_fields_embedding)
        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)
        else:
            token_fields = None
        token_fields_embedding = self.embed_token_fields(token_fields)
        if token_fields_embedding is not None:
            total_fields_embedding.append(token_fields_embedding)
        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)
        if token_seq_fields_embedding is not None:
            total_fields_embedding.append(token_seq_fields_embedding)
        return torch.sum(torch.cat(total_fields_embedding, dim=1), dim=1) + self.bias


class AttLayer(nn.Module):
    """Calculate the attention signal(weight) according the input tensor.

    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].

    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].
    """

    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_signal = self.w(infeatures)
        att_signal = fn.relu(att_signal)
        att_signal = torch.mul(att_signal, self.h)
        att_signal = torch.sum(att_signal, dim=2)
        att_signal = fn.softmax(att_signal, dim=1)
        return att_signal


class Dice(nn.Module):
    """Dice activation function

    .. math::
        f(s)=p(s) \\cdot s+(1-p(s)) \\cdot \\alpha s

    .. math::
        p(s)=\\frac{1} {1 + e^{-\\frac{s-E[s]} {\\sqrt {Var[s] + \\epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha
        score_p = self.sigmoid(score)
        return self.alpha * (1 - score_p) * score + score_p * score


def activation_layer(activation_name='relu', emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'dice':
            activation = Dice(emb_dim)
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError('activation function {} is not implemented'.format(activation_name))
    return activation


class MLPLayers(nn.Module):
    """MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \\*, :math:`H_{in}`) where \\* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \\*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout=0.0, activation='relu', bn=False, init_method=None, last_activation=True):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


def xavier_normal_initialization(module):
    """using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class BaseFactorizationMachine(nn.Module):
    """Calculate FM result over the embeddings

    Args:
        reduce_sum: bool, whether to sum the result, default is True.

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x ** 2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class EulerInteractionLayer(nn.Module):
    """Euler interaction layer is the core component of EulerNet,
    which enables the adaptive learning of explicit feature interactions. An Euler
    interaction layer performs the feature interaction under the complex space one time,
    taking as input a complex representation and outputting a transformed complex representation.
    """

    def __init__(self, config, inshape, outshape):
        super().__init__()
        self.feature_dim = config.embedding_size
        self.apply_norm = config.apply_norm
        init_orders = torch.softmax(torch.randn(inshape // self.feature_dim, outshape // self.feature_dim) / 0.01, dim=0)
        self.inter_orders = nn.Parameter(init_orders)
        self.im = nn.Linear(inshape, outshape)
        self.bias_lam = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01)
        self.bias_theta = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01)
        nn.init.normal_(self.im.weight, mean=0, std=0.1)
        self.drop_ex = nn.Dropout(p=config.drop_ex)
        self.drop_im = nn.Dropout(p=config.drop_im)
        self.norm_r = nn.LayerNorm([self.feature_dim])
        self.norm_p = nn.LayerNorm([self.feature_dim])

    def forward(self, complex_features):
        r, p = complex_features
        lam = r ** 2 + p ** 2 + 1e-08
        theta = torch.atan2(p, r)
        lam, theta = lam.reshape(lam.shape[0], -1, self.feature_dim), theta.reshape(theta.shape[0], -1, self.feature_dim)
        r, p = self.drop_im(r), self.drop_im(p)
        lam = 0.5 * torch.log(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)
        lam, theta = self.drop_ex(lam), self.drop_ex(theta)
        lam, theta = lam @ self.inter_orders + self.bias_lam, theta @ self.inter_orders + self.bias_theta
        lam = torch.exp(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)
        r, p = r.reshape(r.shape[0], -1), p.reshape(p.shape[0], -1)
        r, p = self.im(r), self.im(p)
        r, p = torch.relu(r), torch.relu(p)
        r, p = r.reshape(r.shape[0], -1, self.feature_dim), p.reshape(p.shape[0], -1, self.feature_dim)
        o_r, o_p = r + lam * torch.cos(theta), p + lam * torch.sin(theta)
        o_r, o_p = o_r.reshape(o_r.shape[0], -1, self.feature_dim), o_p.reshape(o_p.shape[0], -1, self.feature_dim)
        if self.apply_norm:
            o_r, o_p = self.norm_r(o_r), self.norm_p(o_p)
        return o_r, o_p


class FieldAwareFactorizationMachine(nn.Module):
    """This is Field-Aware Factorization Machine Module for FFM."""

    def __init__(self, feature_names, feature_dims, feature2id, feature2field, num_fields, embed_dim, device):
        super(FieldAwareFactorizationMachine, self).__init__()
        self.token_feature_names = feature_names[0]
        self.float_feature_names = feature_names[1]
        self.token_seq_feature_names = feature_names[2]
        self.float_seq_feature_names = feature_names[3]
        self.token_feature_dims = feature_dims[0]
        self.float_feature_dims = feature_dims[1]
        self.token_seq_feature_dims = feature_dims[2]
        self.float_seq_feature_dims = feature_dims[3]
        self.feature2id = feature2id
        self.feature2field = feature2field
        self.num_features = len(self.token_feature_names) + len(self.float_feature_names) + len(self.token_seq_feature_names) + len(self.float_seq_feature_names)
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.device = device
        if len(self.token_feature_names) > 0:
            self.num_token_features = len(self.token_feature_names)
            self.token_embeddings = torch.nn.ModuleList([nn.Embedding(sum(self.token_feature_dims), self.embed_dim) for _ in range(self.num_fields)])
            self.token_offsets = np.array((0, *np.cumsum(self.token_feature_dims)[:-1]), dtype=np.long)
            for embedding in self.token_embeddings:
                nn.init.xavier_uniform_(embedding.weight.data)
        if len(self.float_feature_names) > 0:
            self.num_float_features = len(self.float_feature_names)
            self.float_offsets = np.array((0, *np.cumsum(self.float_feature_dims)[:-1]), dtype=np.long)
            self.float_embeddings = torch.nn.ModuleList([nn.Embedding(sum(self.float_feature_dims), self.embed_dim) for _ in range(self.num_fields)])
            for embedding in self.float_embeddings:
                nn.init.xavier_uniform_(embedding.weight.data)
        if len(self.token_seq_feature_names) > 0:
            self.num_token_seq_features = len(self.token_seq_feature_names)
            self.token_seq_embeddings = torch.nn.ModuleList()
            self.token_seq_embedding = torch.nn.ModuleList()
            for i in range(self.num_fields):
                for token_seq_feature_dim in self.token_seq_feature_dims:
                    self.token_seq_embedding.append(nn.Embedding(token_seq_feature_dim, self.embed_dim))
                for embedding in self.token_seq_embedding:
                    nn.init.xavier_uniform_(embedding.weight.data)
                self.token_seq_embeddings.append(self.token_seq_embedding)
        if len(self.float_seq_feature_names) > 0:
            self.num_float_seq_features = len(self.float_seq_feature_names)
            self.float_seq_embeddings = torch.nn.ModuleList()
            self.float_seq_embedding = torch.nn.ModuleList()
            for i in range(self.num_fields):
                for float_seq_feature_dim in self.float_seq_feature_dims:
                    self.float_seq_embedding.append(nn.Embedding(float_seq_feature_dim, self.embed_dim))
                for embedding in self.float_seq_embedding:
                    nn.init.xavier_uniform_(embedding.weight.data)
                self.float_seq_embeddings.append(self.float_seq_embedding)

    def forward(self, input_x):
        """Model the different interaction strengths of different field pairs.


        Args:
            input_x (a tuple): (token_ffm_input, float_ffm_input, token_seq_ffm_input)

                    token_ffm_input (torch.cuda.FloatTensor): [batch_size, num_token_features] or None

                    float_ffm_input (torch.cuda.FloatTensor): [batch_size, num_float_features] or None

                    token_seq_ffm_input (list): length is num_token_seq_features or 0

        Returns:
            torch.cuda.FloatTensor: The results of all features' field-aware interactions.
            shape: [batch_size, num_fields, emb_dim]
        """
        token_ffm_input, float_ffm_input, token_seq_ffm_input, float_seq_ffm_input = input_x[0], input_x[1], input_x[2], input_x[3]
        token_input_x_emb = self._emb_token_ffm_input(token_ffm_input)
        float_input_x_emb = self._emb_float_ffm_input(float_ffm_input)
        token_seq_input_x_emb = self._emb_token_seq_ffm_input(token_seq_ffm_input)
        float_seq_input_x_emb = self._emb_float_seq_ffm_input(float_seq_ffm_input)
        input_x_emb = self._get_input_x_emb(token_input_x_emb, float_input_x_emb, token_seq_input_x_emb, float_seq_input_x_emb)
        output = list()
        for i in range(self.num_features - 1):
            for j in range(i + 1, self.num_features):
                output.append(input_x_emb[self.feature2field[j]][:, i] * input_x_emb[self.feature2field[i]][:, j])
        output = torch.stack(output, dim=1)
        return output

    def _get_input_x_emb(self, token_input_x_emb, float_input_x_emb, token_seq_input_x_emb, float_seq_input_x_emb):
        input_x_emb = []
        zip_args = []
        if len(self.token_feature_names) > 0:
            zip_args.append(token_input_x_emb)
        if len(self.float_feature_names) > 0:
            zip_args.append(float_input_x_emb)
        if len(self.token_seq_feature_names) > 0:
            zip_args.append(token_seq_input_x_emb)
        if len(self.float_seq_feature_names) > 0:
            zip_args.append(float_seq_input_x_emb)
        for tensors in zip(*zip_args):
            input_x_emb.append(torch.cat(tensors, dim=1))
        return input_x_emb

    def _emb_token_ffm_input(self, token_ffm_input):
        token_input_x_emb = []
        if len(self.token_feature_names) > 0:
            token_input_x = token_ffm_input + token_ffm_input.new_tensor(self.token_offsets).unsqueeze(0)
            token_input_x_emb = [self.token_embeddings[i](token_input_x) for i in range(self.num_fields)]
        return token_input_x_emb

    def _emb_float_ffm_input(self, float_ffm_input):
        float_input_x_emb = []
        if len(self.float_feature_names) > 0:
            base, index = torch.split(float_ffm_input, [1, 1], dim=-1)
            index = index.squeeze(-1).long()
            index = index + index.new_tensor(self.float_offsets).unsqueeze(0)
            float_input_x_emb = [(self.float_embeddings[i](index) * base) for i in range(self.num_fields)]
        return float_input_x_emb

    def _emb_token_seq_ffm_input(self, token_seq_ffm_input):
        token_seq_input_x_emb = []
        if len(self.token_seq_feature_names) > 0:
            for i in range(self.num_fields):
                token_seq_result = []
                for j, token_seq in enumerate(token_seq_ffm_input):
                    embedding_table = self.token_seq_embeddings[i][j]
                    mask = token_seq != 0
                    mask = mask.float()
                    value_cnt = torch.sum(mask, dim=1, keepdim=True)
                    token_seq_embedding = embedding_table(token_seq)
                    mask = mask.unsqueeze(2).expand_as(token_seq_embedding)
                    masked_token_seq_embedding = token_seq_embedding * mask.float()
                    result = torch.sum(masked_token_seq_embedding, dim=1)
                    eps = torch.FloatTensor([1e-08])
                    result = torch.div(result, value_cnt + eps)
                    result = result.unsqueeze(1)
                    token_seq_result.append(result)
                token_seq_input_x_emb.append(torch.cat(token_seq_result, dim=1))
        return token_seq_input_x_emb

    def _emb_float_seq_ffm_input(self, float_seq_ffm_input):
        float_seq_input_x_emb = []
        if len(self.float_seq_feature_names) > 0:
            for i in range(self.num_fields):
                float_seq_result = []
                for j, float_seq in enumerate(float_seq_ffm_input):
                    embedding_table = self.float_seq_embeddings[i][j]
                    base, index = torch.split(float_seq, [1, 1], dim=-1)
                    index = index.squeeze(-1)
                    mask = index != 0
                    mask = mask.float()
                    value_cnt = torch.sum(mask, dim=1, keepdim=True)
                    float_seq_embedding = base * embedding_table(index.long())
                    mask = mask.unsqueeze(2).expand_as(float_seq_embedding)
                    masked_float_seq_embedding = float_seq_embedding * mask.float()
                    result = torch.sum(masked_float_seq_embedding, dim=1)
                    eps = torch.FloatTensor([1e-08])
                    result = torch.div(result, value_cnt + eps)
                    result = result.unsqueeze(1)
                    float_seq_result.append(result)
                float_seq_input_x_emb.append(torch.cat(float_seq_result, dim=1))
        return float_seq_input_x_emb


class GraphLayer(nn.Module):
    """
    The implementations of the GraphLayer part and the Attentional Edge Weights part are adapted from https://github.com/xue-pai/FuxiCTR.
    """

    def __init__(self, num_fields, embedding_size):
        super(GraphLayer, self).__init__()
        self.W_in = nn.Parameter(torch.Tensor(num_fields, embedding_size, embedding_size))
        self.W_out = nn.Parameter(torch.Tensor(num_fields, embedding_size, embedding_size))
        xavier_normal_(self.W_in)
        xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_size))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class DAGFM(nn.Module):

    def __init__(self, config):
        super(DAGFM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.type = config['type']
        self.depth = config['depth']
        field_num = config['feature_num']
        embedding_size = config['embedding_size']
        if self.type == 'inner':
            self.p = nn.ParameterList([nn.Parameter(torch.randn(field_num, field_num, embedding_size)) for _ in range(self.depth)])
            for _ in range(self.depth):
                xavier_normal_(self.p[_], gain=1.414)
        elif self.type == 'outer':
            self.p = nn.ParameterList([nn.Parameter(torch.randn(field_num, field_num, embedding_size)) for _ in range(self.depth)])
            self.q = nn.ParameterList([nn.Parameter(torch.randn(field_num, field_num, embedding_size)) for _ in range(self.depth)])
            for _ in range(self.depth):
                xavier_normal_(self.p[_], gain=1.414)
                xavier_normal_(self.q[_], gain=1.414)
        self.adj_matrix = torch.zeros(field_num, field_num, embedding_size)
        for i in range(field_num):
            for j in range(i, field_num):
                self.adj_matrix[i, j, :] += 1
        self.connect_layer = nn.Parameter(torch.eye(field_num).float())
        self.linear = nn.Linear(field_num * (self.depth + 1), 1)

    def FeatureInteraction(self, feature):
        init_state = self.connect_layer @ feature
        h0, ht = init_state, init_state
        state = [torch.sum(init_state, dim=-1)]
        for i in range(self.depth):
            if self.type == 'inner':
                aggr = torch.einsum('bfd,fsd->bsd', ht, self.p[i] * self.adj_matrix)
                ht = h0 * aggr
            elif self.type == 'outer':
                term = torch.einsum('bfd,fsd->bfs', ht, self.p[i] * self.adj_matrix)
                aggr = torch.einsum('bfs,fsd->bsd', term, self.q[i])
                ht = h0 * aggr
            state.append(torch.sum(ht, dim=-1))
        state = torch.cat(state, dim=-1)
        self.logits = self.linear(state)
        self.outputs = torch.sigmoid(self.logits)
        return self.outputs


class CrossNet(nn.Module):

    def __init__(self, config):
        super(CrossNet, self).__init__()
        self.depth = config['depth']
        self.embedding_size = config['embedding_size']
        self.feature_num = config['feature_num']
        self.in_feature_num = self.feature_num * self.embedding_size
        self.cross_layer_w = nn.ParameterList(nn.Parameter(torch.randn(self.in_feature_num, self.in_feature_num)) for _ in range(self.depth))
        self.bias = nn.ParameterList(nn.Parameter(torch.zeros(self.in_feature_num, 1)) for _ in range(self.depth))
        self.linear = nn.Linear(self.in_feature_num, 1)
        nn.init.normal_(self.linear.weight)

    def FeatureInteraction(self, x_0):
        x_0 = x_0.reshape(x_0.shape[0], -1)
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0
        for i in range(self.depth):
            xl_w = torch.matmul(self.cross_layer_w[i], x_l)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x_0, xl_w)
            x_l = xl_dot + x_l
        x_l = x_l.squeeze(dim=2)
        self.logits = self.linear(x_l)
        self.outputs = torch.sigmoid(self.logits)
        return self.outputs

    def forward(self, feature):
        return self.FeatureInteraction(feature)


class CINComp(nn.Module):

    def __init__(self, indim, outdim, config):
        super(CINComp, self).__init__()
        basedim = config['feature_num']
        self.conv = nn.Conv1d(indim * basedim, outdim, 1)

    def forward(self, feature, base):
        return self.conv((feature[:, :, None, :] * base[:, None, :, :]).reshape(feature.shape[0], feature.shape[1] * base.shape[1], -1))


class CIN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.cinlist = [config['feature_num']] + config['cin']
        self.cin = nn.ModuleList([CINComp(self.cinlist[i], self.cinlist[i + 1], config) for i in range(0, len(self.cinlist) - 1)])
        self.linear = nn.Parameter(torch.zeros(sum(self.cinlist) - self.cinlist[0], 1))
        nn.init.normal_(self.linear, mean=0, std=0.01)
        self.backbone = ['cin', 'linear']
        self.loss_fn = nn.BCELoss()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def FeatureInteraction(self, feature):
        base = feature
        x = feature
        p = []
        for comp in self.cin:
            x = comp(x, base)
            p.append(torch.sum(x, dim=-1))
        p = torch.cat(p, dim=-1)
        self.logits = p @ self.linear
        self.outputs = torch.sigmoid(self.logits)
        return self.outputs

    def forward(self, feature):
        return self.FeatureInteraction(feature)


class InnerProductLayer(nn.Module):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.

    """

    def __init__(self, num_feature_field, device):
        """
        Args:
            num_feature_field(int) :number of feature fields.
            device(torch.device) : device object of the model.
        """
        super(InnerProductLayer, self).__init__()
        self.num_feature_field = num_feature_field
        self

    def forward(self, feat_emb):
        """
        Args:
            feat_emb(torch.FloatTensor) :3D tensor with shape: [batch_size,num_pairs,embedding_size].

        Returns:
            inner_product(torch.FloatTensor): The inner product of input tensor. shape of [batch_size, num_pairs]
        """
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]
        q = feat_emb[:, col]
        inner_product = p * q
        return inner_product.sum(dim=-1)


class OuterProductLayer(nn.Module):
    """OuterProduct Layer used in PNN. This implementation is
    adapted from code that the author of the paper published on https://github.com/Atomu2014/product-nets.
    """

    def __init__(self, num_feature_field, embedding_size, device):
        """
        Args:
            num_feature_field(int) :number of feature fields.
            embedding_size(int) :number of embedding size.
            device(torch.device) : device object of the model.
        """
        super(OuterProductLayer, self).__init__()
        self.num_feature_field = num_feature_field
        num_pairs = int(num_feature_field * (num_feature_field - 1) / 2)
        embed_size = embedding_size
        self.kernel = nn.Parameter(torch.rand(embed_size, num_pairs, embed_size), requires_grad=True)
        nn.init.xavier_uniform_(self.kernel)
        self

    def forward(self, feat_emb):
        """
        Args:
            feat_emb(torch.FloatTensor) :3D tensor with shape: [batch_size,num_pairs,embedding_size].

        Returns:
            outer_product(torch.FloatTensor): The outer product of input tensor. shape of [batch_size, num_pairs]
        """
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]
        q = feat_emb[:, col]
        p.unsqueeze_(dim=1)
        p = torch.mul(p, self.kernel.unsqueeze(0))
        p = torch.sum(p, dim=-1)
        p = torch.transpose(p, 2, 1)
        outer_product = p * q
        return outer_product.sum(dim=-1)


def add_noise(t, mag=1e-05):
    return t + mag * torch.rand(t.shape)


def soft_threshold(x, threshold):
    return (np.abs(x) > threshold) * (np.abs(x) - threshold) * np.sign(x)


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class AutoEncoderMixin(object):
    """This is a common part of auto-encoders. All the auto-encoder models should inherit this class,
    including CDAE, MacridVAE, MultiDAE, MultiVAE, RaCT and RecVAE.
    The base AutoEncoderMixin class provides basic dataset information and rating matrix function.
    """

    def build_histroy_items(self, dataset):
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id
        self.history_item_value = self.history_item_value

    def get_rating_matrix(self, user):
        """Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).repeat_interleave(self.history_item_id.shape[1], dim=0)
        rating_matrix = torch.zeros(1, device=self.device).repeat(user.shape[0], self.n_items)
        rating_matrix.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        return rating_matrix


class ConvNCFBPRLoss(nn.Module):
    """ConvNCFBPRLoss, based on Bayesian Personalized Ranking,

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = ConvNCFBPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self):
        super(ConvNCFBPRLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        distance = pos_score - neg_score
        loss = torch.sum(torch.log(1 + torch.exp(-distance)))
        return loss


class CNNLayers(nn.Module):
    """CNNLayers

    Args:
        - channels(list): a list contains the channels of each layer in cnn layers
        - kernel(list): a list contains the kernels of each layer in cnn layers
        - strides(list): a list contains the channels of each layer in cnn layers
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'
                      candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

        .. math::
            H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[0] - \\text{dilation}[0]
                      \\times (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

        .. math::
            W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[1] - \\text{dilation}[1]
                      \\times (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

    Examples::

        >>> m = CNNLayers([1, 32, 32], [2,2], [2,2], 'relu')
        >>> input = torch.randn(128, 1, 64, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 32, 16, 16])
    """

    def __init__(self, channels, kernels, strides, activation='relu', init_method=None):
        super(CNNLayers, self).__init__()
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.activation = activation
        self.init_method = init_method
        self.num_of_nets = len(self.channels) - 1
        if len(kernels) != len(strides) or self.num_of_nets != len(kernels):
            raise RuntimeError("channels, kernels and strides don't match\n")
        cnn_modules = []
        for i in range(self.num_of_nets):
            cnn_modules.append(nn.Conv2d(self.channels[i], self.channels[i + 1], self.kernels[i], stride=self.strides[i]))
            if self.activation.lower() == 'sigmoid':
                cnn_modules.append(nn.Sigmoid())
            elif self.activation.lower() == 'tanh':
                cnn_modules.append(nn.Tanh())
            elif self.activation.lower() == 'relu':
                cnn_modules.append(nn.ReLU())
            elif self.activation.lower() == 'leakyrelu':
                cnn_modules.append(nn.LeakyReLU())
            elif self.activation.lower() == 'none':
                pass
        self.cnn_layers = nn.Sequential(*cnn_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.cnn_layers(input_feature)


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


def sample_cor_samples(n_users, n_items, cor_batch_size):
    """This is a function that sample item ids and user ids.

    Args:
        n_users (int): number of users in total
        n_items (int): number of items in total
        cor_batch_size (int): number of id to sample

    Returns:
        list: cor_users, cor_items. The result sampled ids with both as cor_batch_size long.

    Note:
        We have to sample some embedded representations out of all nodes.
        Because we have no way to store cor-distance for each pair.
    """
    cor_users = rd.sample(list(range(n_users)), cor_batch_size)
    cor_items = rd.sample(list(range(n_items)), cor_batch_size)
    return cor_users, cor_items


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional. (N,)
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, dims: 'typing.List', emb_size: 'int', time_type='cat', act_func='tanh', norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.dims = dims
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        if self.time_type == 'cat':
            self.dims[0] += self.time_emb_dim
        else:
            raise ValueError('Unimplemented timestep embedding type %s' % self.time_type)
        self.mlp_layers = MLPLayers(layers=self.dims, dropout=0, activation=act_func, last_activation=False)
        self.drop = nn.Dropout(dropout)
        self.apply(xavier_normal_initialization)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        h = self.mlp_layers(h)
        return h


class ModelMeanType(enum.Enum):
    START_X = enum.auto()
    EPSILON = enum.auto()


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    Args:
        num_diffusion_timesteps (int): the number of betas to produce.
        alpha_bar (Callable): a lambda that takes an argument t from 0 to 1 and
                   produces the cumulative product of (1-beta) up to that
                   part of the diffusion process.
        max_beta (int): the maximum beta to use; use values lower than 1 to
                  prevent singularities.
    Returns:
        np.ndarray: a 1-D array of beta values.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def orthogonal(shape, scale=1.1):
    """
    Initialization function for weights in class GCMC.
    From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    flat_shape = shape[0], np.prod(shape[1:])
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return torch.tensor(scale * q[:shape[0], :shape[1]], dtype=torch.float32)


class BiDecoder(nn.Module):
    """Bi-linear decoder
    BiDecoder takes pairs of node embeddings and predicts respective entries in the adjacency matrix.
    """

    def __init__(self, input_dim, output_dim, drop_prob, device, num_weights=3, act=lambda x: x):
        super(BiDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_weights = num_weights
        self.device = device
        self.activate = act
        self.dropout_prob = drop_prob
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.weights = nn.ParameterList([nn.Parameter(orthogonal([self.input_dim, self.input_dim])) for _ in range(self.num_weights)])
        self.dense_layer = nn.Linear(self.num_weights, self.output_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        dense_init_range = math.sqrt(self.output_dim / (self.num_weights + self.output_dim))
        self.dense_layer.weight.data.uniform_(-dense_init_range, dense_init_range)

    def forward(self, u_inputs, i_inputs, users, items=None):
        u_inputs = self.dropout(u_inputs)
        i_inputs = self.dropout(i_inputs)
        if items is not None:
            users_emb = u_inputs[users]
            items_emb = i_inputs[items]
            basis_outputs = []
            for i in range(self.num_weights):
                users_emb_temp = torch.mm(users_emb, self.weights[i])
                scores = torch.mul(users_emb_temp, items_emb)
                scores = torch.sum(scores, dim=1)
                basis_outputs.append(scores)
        else:
            users_emb = u_inputs[users]
            items_emb = i_inputs
            basis_outputs = []
            for i in range(self.num_weights):
                users_emb_temp = torch.mm(users_emb, self.weights[i])
                scores = torch.mm(users_emb_temp, items_emb.transpose(0, 1))
                basis_outputs.append(scores.view(-1))
        basis_outputs = torch.stack(basis_outputs, dim=1)
        basis_outputs = self.dense_layer(basis_outputs)
        output = self.activate(basis_outputs)
        return output


class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x
        mask = (torch.rand(x._values().size()) + self.kprob).floor().type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


class GcEncoder(nn.Module):
    """Graph Convolutional Encoder
    GcEncoder take as input an :math:`N \\times D` feature matrix :math:`X` and a graph adjacency matrix :math:`A`,
    and produce an :math:`N \\times E` node embedding matrix;
    Note that :math:`N` denotes the number of nodes, :math:`D` the number of input features,
    and :math:`E` the embedding size.
    """

    def __init__(self, accum, num_user, num_item, support, input_dim, gcn_output_dim, dense_output_dim, drop_prob, device, sparse_feature=True, act_dense=lambda x: x, share_user_item_weights=True, bias=False):
        super(GcEncoder, self).__init__()
        self.num_users = num_user
        self.num_items = num_item
        self.input_dim = input_dim
        self.gcn_output_dim = gcn_output_dim
        self.dense_output_dim = dense_output_dim
        self.accum = accum
        self.sparse_feature = sparse_feature
        self.device = device
        self.dropout_prob = drop_prob
        self.dropout = nn.Dropout(p=self.dropout_prob)
        if self.sparse_feature:
            self.sparse_dropout = SparseDropout(p=self.dropout_prob)
        else:
            self.sparse_dropout = nn.Dropout(p=self.dropout_prob)
        self.dense_activate = act_dense
        self.activate = nn.ReLU()
        self.share_weights = share_user_item_weights
        self.bias = bias
        self.support = support
        self.num_support = len(support)
        if self.accum == 'sum':
            self.weights_u = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.input_dim, self.gcn_output_dim), requires_grad=True) for _ in range(self.num_support)])
            if share_user_item_weights:
                self.weights_v = self.weights_u
            else:
                self.weights_v = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.input_dim, self.gcn_output_dim), requires_grad=True) for _ in range(self.num_support)])
        else:
            assert self.gcn_output_dim % self.num_support == 0, 'output_dim must be multiple of num_support for stackGC'
            self.sub_hidden_dim = self.gcn_output_dim // self.num_support
            self.weights_u = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.input_dim, self.sub_hidden_dim), requires_grad=True) for _ in range(self.num_support)])
            if share_user_item_weights:
                self.weights_v = self.weights_u
            else:
                self.weights_v = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.input_dim, self.sub_hidden_dim), requires_grad=True) for _ in range(self.num_support)])
        self.dense_layer_u = nn.Linear(self.gcn_output_dim, self.dense_output_dim, bias=self.bias)
        if share_user_item_weights:
            self.dense_layer_v = self.dense_layer_u
        else:
            self.dense_layer_v = nn.Linear(self.gcn_output_dim, self.dense_output_dim, bias=self.bias)
        self._init_weights()

    def _init_weights(self):
        init_range = math.sqrt((self.num_support + 1) / (self.input_dim + self.gcn_output_dim))
        for w in range(self.num_support):
            self.weights_u[w].data.uniform_(-init_range, init_range)
        if not self.share_weights:
            for w in range(self.num_support):
                self.weights_v[w].data.uniform_(-init_range, init_range)
        dense_init_range = math.sqrt((self.num_support + 1) / (self.dense_output_dim + self.gcn_output_dim))
        self.dense_layer_u.weight.data.uniform_(-dense_init_range, dense_init_range)
        if not self.share_weights:
            self.dense_layer_v.weight.data.uniform_(-dense_init_range, dense_init_range)
        if self.bias:
            self.dense_layer_u.bias.data.fill_(0)
            if not self.share_weights:
                self.dense_layer_v.bias.data.fill_(0)

    def forward(self, user_X, item_X):
        user_X = self.sparse_dropout(user_X)
        item_X = self.sparse_dropout(item_X)
        embeddings = []
        if self.accum == 'sum':
            wu = 0.0
            wv = 0.0
            for i in range(self.num_support):
                wu = self.weights_u[i] + wu
                wv = self.weights_v[i] + wv
                if self.sparse_feature:
                    temp_u = torch.sparse.mm(user_X, wu)
                    temp_v = torch.sparse.mm(item_X, wv)
                else:
                    temp_u = torch.mm(user_X, wu)
                    temp_v = torch.mm(item_X, wv)
                all_embedding = torch.cat([temp_u, temp_v])
                graph_A = self.support[i]
                all_emb = torch.sparse.mm(graph_A, all_embedding)
                embeddings.append(all_emb)
            embeddings = torch.stack(embeddings, dim=1)
            embeddings = torch.sum(embeddings, dim=1)
        else:
            for i in range(self.num_support):
                if self.sparse_feature:
                    temp_u = torch.sparse.mm(user_X, self.weights_u[i])
                    temp_v = torch.sparse.mm(item_X, self.weights_v[i])
                else:
                    temp_u = torch.mm(user_X, self.weights_u[i])
                    temp_v = torch.mm(item_X, self.weights_v[i])
                all_embedding = torch.cat([temp_u, temp_v])
                graph_A = self.support[i]
                all_emb = torch.sparse.mm(graph_A, all_embedding)
                embeddings.append(all_emb)
            embeddings = torch.cat(embeddings, dim=1)
        users, items = torch.split(embeddings, [self.num_users, self.num_items])
        u_hidden = self.activate(users)
        v_hidden = self.activate(items)
        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)
        u_hidden = self.dense_layer_u(u_hidden)
        v_hidden = self.dense_layer_v(v_hidden)
        u_outputs = self.dense_activate(u_hidden)
        v_outputs = self.dense_activate(v_hidden)
        return u_outputs, v_outputs


class ComputeSimilarity:

    def __init__(self, dataMatrix, topk=100, shrink=0, normalize=True):
        """Computes the cosine similarity of dataMatrix

        If it is computed on :math:`URM=|users| \\times |items|`, pass the URM.

        If it is computed on :math:`ICM=|items| \\times |features|`, pass the ICM transposed.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink (int) :  hyper-parameter in calculate cosine distance.
            normalize (bool):   If True divide the dot product by the product of the norms.
        """
        super(ComputeSimilarity, self).__init__()
        self.shrink = shrink
        self.normalize = normalize
        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topk, self.n_columns)
        self.dataMatrix = dataMatrix.copy()

    def compute_similarity(self, method, block_size=100):
        """Compute the similarity for the given dataset

        Args:
            method (str) : Caculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
            block_size (int): divide matrix to :math:`n\\_rows \\div block\\_size` to calculate cosine_distance if method is 'user',
                 otherwise, divide matrix to :math:`n\\_columns \\div block\\_size`.

        Returns:

            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num],
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """
        values = []
        rows = []
        cols = []
        neigh = []
        self.dataMatrix = self.dataMatrix.astype(np.float32)
        if method == 'user':
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
            end_local = self.n_rows
        elif method == 'item':
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError("Make sure 'method' in ['user', 'item']!")
        sumOfSquared = np.sqrt(sumOfSquared)
        start_block = 0
        while start_block < end_local:
            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block
            if method == 'user':
                data = self.dataMatrix[start_block:end_block, :]
            else:
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray()
            if method == 'user':
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)
            for index_in_block in range(this_block_size):
                this_line_weights = this_block_weights[:, index_in_block]
                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0
                if self.normalize:
                    denominator = sumOfSquared[Index] * sumOfSquared + self.shrink + 1e-06
                    this_line_weights = np.multiply(this_line_weights, 1 / denominator)
                elif self.shrink != 0:
                    this_line_weights = this_line_weights / self.shrink
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_partition_sorting = np.argsort(-this_line_weights[relevant_partition])
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)
                values.extend(this_line_weights[top_k_idx][notZerosMask])
                if method == 'user':
                    rows.extend(np.ones(numNotZeros) * Index)
                    cols.extend(top_k_idx[notZerosMask])
                else:
                    rows.extend(top_k_idx[notZerosMask])
                    cols.extend(np.ones(numNotZeros) * Index)
            start_block += block_size
        if method == 'user':
            W_sparse = sp.csr_matrix((values, (rows, cols)), shape=(self.n_rows, self.n_rows), dtype=np.float32)
        else:
            W_sparse = sp.csr_matrix((values, (rows, cols)), shape=(self.n_columns, self.n_columns), dtype=np.float32)
        return neigh, W_sparse.tocsc()


class AutoEncoder(nn.Module):
    """
    Guassian Diffusion for large-scale recommendation.
    """

    def __init__(self, item_emb, n_cate, in_dims, out_dims, device, act_func, reparam=True, dropout=0.1):
        super(AutoEncoder, self).__init__()
        self.item_emb = item_emb
        self.n_cate = n_cate
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.act_func = act_func
        self.n_item = len(item_emb)
        self.reparam = reparam
        self.dropout = nn.Dropout(dropout)
        if n_cate == 1:
            in_dims_temp = [self.n_item + 1] + self.in_dims[:-1] + [self.in_dims[-1] * 2]
            out_dims_temp = [self.in_dims[-1]] + self.out_dims + [self.n_item + 1]
            self.encoder = MLPLayers(in_dims_temp, activation=self.act_func)
            self.decoder = MLPLayers(out_dims_temp, activation=self.act_func, last_activation=False)
        else:
            self.cluster_ids, _ = kmeans(X=item_emb, num_clusters=n_cate, distance='euclidean', device=device)
            category_idx = []
            for i in range(n_cate):
                idx = np.argwhere(self.cluster_ids.numpy() == i).flatten().tolist()
                category_idx.append(torch.tensor(idx, dtype=int) + 1)
            self.category_idx = category_idx
            self.category_map = torch.cat(tuple(category_idx), dim=-1)
            self.category_len = [len(self.category_idx[i]) for i in range(n_cate)]
            None
            assert sum(self.category_len) == self.n_item
            encoders = []
            decode_dim = []
            for i in range(n_cate):
                if i == n_cate - 1:
                    latent_dims = list(self.in_dims - np.array(decode_dim).sum(axis=0))
                else:
                    latent_dims = [int(self.category_len[i] / self.n_item * self.in_dims[j]) for j in range(len(self.in_dims))]
                    latent_dims = [(latent_dims[j] if latent_dims[j] != 0 else 1) for j in range(len(self.in_dims))]
                in_dims_temp = [self.category_len[i]] + latent_dims[:-1] + [latent_dims[-1] * 2]
                encoders.append(MLPLayers(in_dims_temp, activation=self.act_func))
                decode_dim.append(latent_dims)
            self.encoder = nn.ModuleList(encoders)
            None
            self.decode_dim = [decode_dim[i][::-1] for i in range(len(decode_dim))]
            if len(out_dims) == 0:
                out_dim = self.in_dims[-1]
                self.decoder = MLPLayers([out_dim, self.n_item], activation=None)
            else:
                decoders = []
                for i in range(n_cate):
                    out_dims_temp = self.decode_dim[i] + [self.category_len[i]]
                    decoders.append(MLPLayers(out_dims_temp, activation=self.act_func, last_activation=False))
                self.decoder = nn.ModuleList(decoders)
        self.apply(xavier_normal_initialization)

    def Encode(self, batch):
        batch = self.dropout(batch)
        if self.n_cate == 1:
            hidden = self.encoder(batch)
            mu = hidden[:, :self.in_dims[-1]]
            logvar = hidden[:, self.in_dims[-1]:]
            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)
            else:
                latent = mu
            kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            return batch, latent, kl_divergence
        else:
            batch_cate = []
            for i in range(self.n_cate):
                batch_cate.append(batch[:, self.category_idx[i]])
            latent_mu = []
            latent_logvar = []
            for i in range(self.n_cate):
                hidden = self.encoder[i](batch_cate[i])
                latent_mu.append(hidden[:, :self.decode_dim[i][0]])
                latent_logvar.append(hidden[:, self.decode_dim[i][0]:])
            mu = torch.cat(tuple(latent_mu), dim=-1)
            logvar = torch.cat(tuple(latent_logvar), dim=-1)
            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)
            else:
                latent = mu
            kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            return torch.cat(tuple(batch_cate), dim=-1), latent, kl_divergence

    def reparamterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def Decode(self, batch):
        if len(self.out_dims) == 0 or self.n_cate == 1:
            return self.decoder(batch)
        else:
            batch_cate = []
            start = 0
            for i in range(self.n_cate):
                end = start + self.decode_dim[i][0]
                batch_cate.append(batch[:, start:end])
                start = end
            pred_cate = []
            for i in range(self.n_cate):
                pred_cate.append(self.decoder[i](batch_cate[i]))
            pred = torch.cat(tuple(pred_cate), dim=-1)
            return pred


def compute_loss(recon_x, x):
    return -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))


def xavier_uniform_initialization(module):
    """using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class NegSamplingLoss(nn.Module):

    def __init__(self):
        super(NegSamplingLoss, self).__init__()

    def forward(self, sign, score):
        return -torch.mean(torch.log(torch.sigmoid(sign * score)))


class BiGNNLayer(nn.Module):
    """Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \\otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, lap_matrix, eye_matrix, features):
        x = torch.sparse.mm(lap_matrix, features)
        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)
        return inter_part1 + inter_part2


def swish(x):
    """Swish activation function:

    .. math::
        \\text{Swish}(x) = \\frac{x}{1 + \\exp(-x)}
    """
    return x.mul(torch.sigmoid(x))


class Encoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim, eps=0.1):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_prob):
        x = F.normalize(x)
        x = F.dropout(x, dropout_prob, training=self.training)
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights):
        super(CompositePrior, self).__init__()
        self.mixture_weights = mixture_weights
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        density_per_gaussian = torch.stack(gaussians, dim=-1)
        return torch.logsumexp(density_per_gaussian, dim=-1)


class InnerProductLoss(nn.Module):
    """This is the inner-product loss used in CFKG for optimization."""

    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        pos_score = torch.mul(anchor, positive).sum(dim=1)
        neg_score = torch.mul(anchor, negative).sum(dim=1)
        return (F.softplus(-pos_score) + F.softplus(neg_score)).mean()


class Aggregator(nn.Module):

    def __init__(self, item_only=False, attention=True):
        super(Aggregator, self).__init__()
        self.item_only = item_only
        self.attention = attention

    def forward(self, entity_emb, user_emb, relation_emb, edge_index, edge_type, inter_matrix):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        if self.attention:
            neigh_relation_emb_weight = self.calculate_sim_hrt(entity_emb[head], entity_emb[tail], edge_relation_emb)
            neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0], neigh_relation_emb.shape[1])
            neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, index=head, dim=0)
            neigh_relation_emb = torch.mul(neigh_relation_emb_weight, neigh_relation_emb)
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        if self.item_only:
            return entity_agg
        user_agg = torch.sparse.mm(inter_matrix, entity_emb)
        score = torch.mm(user_emb, relation_emb.t())
        score = torch.softmax(score, dim=-1)
        user_agg = user_agg + torch.mm(score, relation_emb) * user_agg
        return entity_agg, user_agg

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):
        """
        The calculation method of attention weight here follows the code implementation of the author, which is
        slightly different from that described in the paper.
        """
        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, config, embedding_size, n_relations, edge_index, edge_type, inter_matrix, device):
        super(GraphConv, self).__init__()
        self.n_relations = n_relations
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.inter_matrix = inter_matrix
        self.embedding_size = embedding_size
        self.n_hops = config['n_hops']
        self.node_dropout_rate = config['node_dropout_rate']
        self.mess_dropout_rate = config['mess_dropout_rate']
        self.topk = config['k']
        self.lambda_coeff = config['lambda_coeff']
        self.build_graph_separately = config['build_graph_separately']
        self.device = device
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        if self.build_graph_separately:
            """
            In the original author's implementation(https://github.com/CCIIPLab/MCCLK), the process of constructing
            k-Nearest-Neighbor item-item semantic graph(section 4.1 in paper) and encoding structural view(section 4.3.1 in paper)
            are combined. This implementation improves the computational efficiency, but is slightly different from the
            model structure described in the paper. We use the parameter `build_graph_separately` to control whether to
            use a separate GCN to build a item-item semantic graph. If `build_graph_separately` is set to true, the model
            structure will be the same as that described in the paper. Otherwise, the author's code implementation will be followed.
            """
            self.bg_convs = nn.ModuleList()
            for i in range(self.n_hops):
                self.bg_convs.append(Aggregator(item_only=True, attention=False))
        self.convs = nn.ModuleList()
        for i in range(self.n_hops):
            self.convs.append(Aggregator())
        self.node_dropout = SparseDropout(p=self.mess_dropout_rate)
        self.mess_dropout = nn.Dropout(p=self.mess_dropout_rate)
        self.apply(xavier_normal_initialization)

    def edge_sampling(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def forward(self, user_emb, entity_emb):
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.edge_sampling(self.edge_index, self.edge_type, self.node_dropout_rate)
            inter_matrix = self.node_dropout(self.inter_matrix)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            inter_matrix = self.inter_matrix
        origin_entity_emb = entity_emb
        entity_res_emb = [entity_emb]
        user_res_emb = [user_emb]
        relation_emb = self.relation_embedding.weight
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, relation_emb, edge_index, edge_type, inter_matrix)
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
                user_emb = self.mess_dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            entity_res_emb.append(entity_emb)
            user_res_emb.append(user_emb)
        entity_res_emb = torch.stack(entity_res_emb, dim=1)
        entity_res_emb = entity_res_emb.mean(dim=1, keepdim=False)
        user_res_emb = torch.stack(user_res_emb, dim=1)
        user_res_emb = user_res_emb.mean(dim=1, keepdim=False)
        if self.build_graph_separately:
            item_adj = self._build_graph_separately(origin_entity_emb)
        else:
            origin_item_adj = self.build_adj(origin_entity_emb, self.topk)
            item_adj = (1 - self.lambda_coeff) * self.build_adj(entity_res_emb, self.topk) + self.lambda_coeff * origin_item_adj
        return entity_res_emb, user_res_emb, item_adj

    def build_adj(self, context, topk):
        """Construct a k-Nearest-Neighbor item-item semantic graph.

        Returns:
            Sparse tensor of the normalized item-item matrix.
        """
        n_entities = context.shape[0]
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).cpu()
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_val, knn_index = torch.topk(sim, topk, dim=-1)
        knn_val, knn_index = knn_val, knn_index
        y = knn_index.reshape(-1)
        x = torch.arange(0, n_entities).unsqueeze(dim=-1)
        x = x.expand(n_entities, topk).reshape(-1)
        indice = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0)
        value = knn_val.reshape(-1)
        adj_sparsity = torch.sparse.FloatTensor(indice.data, value.data, torch.Size([n_entities, n_entities]))
        rowsum = torch.sparse.sum(adj_sparsity, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt_value = d_inv_sqrt._values()
        x = torch.arange(0, n_entities).unsqueeze(dim=0)
        x = x.expand(2, n_entities)
        d_mat_inv_sqrt_indice = x
        d_mat_inv_sqrt = torch.sparse.FloatTensor(d_mat_inv_sqrt_indice, d_mat_inv_sqrt_value, torch.Size([n_entities, n_entities]))
        L_norm = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity), d_mat_inv_sqrt)
        return L_norm

    def _build_graph_separately(self, entity_emb):
        if self.node_dropout_rate > 0.0:
            edge_index, edge_type = self.edge_sampling(self.edge_index, self.edge_type, self.node_dropout_rate)
            inter_matrix = self.node_dropout(self.inter_matrix)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            inter_matrix = self.inter_matrix
        origin_item_adj = self.build_adj(entity_emb, self.topk)
        entity_res_emb = [entity_emb]
        relation_emb = self.relation_embedding.weight
        for i in range(len(self.bg_convs)):
            entity_emb = self.bg_convs[i](entity_emb, None, relation_emb, edge_index, edge_type, inter_matrix)
            if self.mess_dropout_rate > 0.0:
                entity_emb = self.mess_dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)
            entity_res_emb.append(entity_emb)
        entity_res_emb = torch.stack(entity_res_emb, dim=1)
        entity_res_emb = entity_res_emb.mean(dim=1, keepdim=False)
        item_adj = (1 - self.lambda_coeff) * self.build_adj(entity_res_emb, self.topk) + self.lambda_coeff * origin_item_adj
        return item_adj


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0)
        cache_zero = torch.tensor(0.0)
        emb_loss = torch.tensor(0.0)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


def alignLoss(emb1, emb2, L1_flag=False):
    if L1_flag:
        distance = torch.sum(torch.abs(emb1 - emb2), 1)
    else:
        distance = torch.sum((emb1 - emb2) ** 2, 1)
    return distance.mean()


def orthogonalLoss(rel_embeddings, norm_embeddings):
    return torch.sum(torch.sum(norm_embeddings * rel_embeddings, dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))


class CrossCompressUnit(nn.Module):
    """This is Cross&Compress Unit for MKR model to model feature interactions between items and entities."""

    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = nn.Linear(dim, 1, bias=True)
        self.fc_ev = nn.Linear(dim, 1, bias=True)
        self.fc_ve = nn.Linear(dim, 1, bias=True)
        self.fc_ee = nn.Linear(dim, 1, bias=True)

    def forward(self, inputs):
        v, e = inputs
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0, 2, 1)
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_output = v_intermediate.view(-1, self.dim)
        e_output = e_intermediate.view(-1, self.dim)
        return v_output, e_output


class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.

    Args:
        queries (torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys (torch.Tensor): user_hist, [B, T, H]
        keys_length (torch.Tensor): mask, [B]

    Returns:
        torch.Tensor: result
    """

    def __init__(self, mask_mat, att_hidden_size=(80, 40), activation='sigmoid', softmax_stag=False, return_seq_weight=True):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.mask_mat = mask_mat
        self.att_mlp_layers = MLPLayers(self.att_hidden_size, activation=self.activation, bn=False)
        self.dense = nn.Linear(self.att_hidden_size[-1], 1)

    def forward(self, queries, keys, keys_length):
        embedding_size = queries.shape[-1]
        hist_len = keys.shape[1]
        queries = queries.repeat(1, hist_len)
        queries = queries.view(-1, hist_len, embedding_size)
        input_tensor = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        output = self.att_mlp_layers(input_tensor)
        output = torch.transpose(self.dense(output), -1, -2)
        output = output.squeeze(1)
        mask = self.mask_mat.repeat(output.size(0), 1)
        mask = mask >= keys_length.unsqueeze(1)
        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0
        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / embedding_size ** 0.5
        if self.softmax_stag:
            output = fn.softmax(output, dim=2)
        if not self.return_seq_weight:
            output = torch.matmul(output, keys)
        return output


class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (hidden_size, n_heads))
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {'gelu': self.gelu, 'relu': fn.relu, 'swish': self.swish, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    """One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(self, n_layers=2, n_heads=2, hidden_size=64, inner_size=256, hidden_dropout_prob=0.5, attn_dropout_prob=0.5, hidden_act='gelu', layer_norm_eps=1e-12):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ItemToInterestAggregation(nn.Module):

    def __init__(self, seq_len, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):
        D_matrix = torch.matmul(input_tensor, self.theta)
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        result = torch.einsum('nij, nik -> nkj', input_tensor, D_matrix)
        return result


class LightMultiHeadAttention(nn.Module):

    def __init__(self, n_heads, k_interests, hidden_size, seq_len, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(LightMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (hidden_size, n_heads))
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.attpooling_key = ItemToInterestAggregation(seq_len, hidden_size, k_interests)
        self.attpooling_value = ItemToInterestAggregation(seq_len, hidden_size, k_interests)
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, pos_emb):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.attpooling_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(self.attpooling_value(mixed_value_layer))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)
        value_layer_pos = self.transpose_for_scores(mixed_value_layer)
        pos_emb = self.pos_ln(pos_emb).unsqueeze(0)
        pos_query_layer = self.transpose_for_scores(self.pos_q_linear(pos_emb)) * self.pos_scaling
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb))
        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        abs_pos_bias = nn.Softmax(dim=-2)(abs_pos_bias)
        context_layer_pos = torch.matmul(abs_pos_bias, value_layer_pos)
        context_layer = context_layer_item + context_layer_pos
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LightTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): the output of the point-wise feed-forward sublayer, is the output of the transformer layer
    """

    def __init__(self, n_heads, k_interests, hidden_size, seq_len, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(LightTransformerLayer, self).__init__()
        self.multi_head_attention = LightMultiHeadAttention(n_heads, k_interests, hidden_size, seq_len, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, pos_emb):
        attention_output = self.multi_head_attention(hidden_states, pos_emb)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LightTransformerEncoder(nn.Module):
    """One LightTransformerEncoder consists of several LightTransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'.
            candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(self, n_layers=2, n_heads=2, k_interests=5, hidden_size=64, seq_len=50, inner_size=256, hidden_dropout_prob=0.5, attn_dropout_prob=0.5, hidden_act='gelu', layer_norm_eps=1e-12):
        super(LightTransformerEncoder, self).__init__()
        layer = LightTransformerLayer(n_heads, k_interests, hidden_size, seq_len, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, pos_emb, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TrandformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer layers' output,
            otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, pos_emb)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ContextSeqEmbAbstractLayer(nn.Module):
    """For Deep Interest Network and feature-rich sequential recommender systems, return features embedding matrices."""

    def __init__(self):
        super(ContextSeqEmbAbstractLayer, self).__init__()
        self.token_field_offsets = {}
        self.float_field_offsets = {}
        self.token_embedding_table = nn.ModuleDict()
        self.float_embedding_table = nn.ModuleDict()
        self.token_seq_embedding_table = nn.ModuleDict()
        self.float_seq_embedding_table = nn.ModuleDict()
        self.token_field_names = None
        self.token_field_dims = None
        self.float_field_names = None
        self.float_field_dims = None
        self.token_seq_field_names = None
        self.token_seq_field_dims = None
        self.float_seq_field_names = None
        self.float_seq_field_dims = None
        self.num_feature_field = None

    def get_fields_name_dim(self):
        """get user feature field and item feature field."""
        self.token_field_names = {type: [] for type in self.types}
        self.token_field_dims = {type: [] for type in self.types}
        self.float_field_names = {type: [] for type in self.types}
        self.float_field_dims = {type: [] for type in self.types}
        self.token_seq_field_names = {type: [] for type in self.types}
        self.token_seq_field_dims = {type: [] for type in self.types}
        self.num_feature_field = {type: (0) for type in self.types}
        self.float_seq_field_names = {type: [] for type in self.types}
        self.float_seq_field_dims = {type: [] for type in self.types}
        for type in self.types:
            for field_name in self.field_names[type]:
                if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.token_field_names[type].append(field_name)
                    self.token_field_dims[type].append(self.dataset.num(field_name))
                elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.token_seq_field_names[type].append(field_name)
                    self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                elif self.dataset.field2type[field_name] == FeatureType.FLOAT and field_name in self.dataset.config['numerical_features']:
                    self.float_field_names[type].append(field_name)
                    self.float_field_dims[type].append(self.dataset.num(field_name))
                elif self.dataset.field2type[field_name] == FeatureType.FLOAT_SEQ and field_name in self.dataset.config['numerical_features']:
                    self.float_seq_field_names[type].append(field_name)
                    self.float_seq_field_dims[type].append(self.dataset.num(field_name))
                else:
                    continue
                self.num_feature_field[type] += 1

    def get_embedding(self):
        """get embedding of all features."""
        for type in self.types:
            if len(self.token_field_dims[type]) > 0:
                self.token_field_offsets[type] = np.array((0, *np.cumsum(self.token_field_dims[type])[:-1]), dtype=np.long)
                self.token_embedding_table[type] = FMEmbedding(self.token_field_dims[type], self.token_field_offsets[type], self.embedding_size)
            if len(self.float_field_dims[type]) > 0:
                self.float_field_offsets[type] = np.array((0, *np.cumsum(self.float_field_dims[type])[:-1]), dtype=np.long)
                self.float_embedding_table[type] = FLEmbedding(self.float_field_dims[type], self.float_field_offsets[type], self.embedding_size)
            if len(self.token_seq_field_dims) > 0:
                self.token_seq_embedding_table[type] = nn.ModuleList()
                for token_seq_field_dim in self.token_seq_field_dims[type]:
                    self.token_seq_embedding_table[type].append(nn.Embedding(token_seq_field_dim, self.embedding_size))
            if len(self.float_seq_field_dims) > 0:
                self.float_seq_embedding_table[type] = nn.ModuleList()
                for float_seq_field_dim in self.float_seq_field_dims[type]:
                    self.float_seq_embedding_table[type].append(nn.Embedding(float_seq_field_dim, self.embedding_size))

    def embed_float_fields(self, float_fields, type, embed=True):
        """Get the embedding of float fields.
        In the following three functions("embed_float_fields" "embed_token_fields" "embed_token_seq_fields")
        when the type is user, [batch_size, max_item_length] should be recognised as [batch_size]

        Args:
            float_fields(torch.Tensor): [batch_size, max_item_length, num_float_field]
            type(str): user or item
            embed(bool): embed or not

        Returns:
            torch.Tensor: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]

        """
        if float_fields is None:
            return None
        if type == 'item':
            embedding_shape = float_fields.shape[:-1] + (-1,)
            float_fields = float_fields.reshape(-1, float_fields.shape[-2], float_fields.shape[-1])
            float_embedding = self.float_embedding_table[type](float_fields)
            float_embedding = float_embedding.view(embedding_shape)
        else:
            float_embedding = self.float_embedding_table[type](float_fields)
        return float_embedding

    def embed_token_fields(self, token_fields, type):
        """Get the embedding of token fields

        Args:
            token_fields(torch.Tensor): input, [batch_size, max_item_length, num_token_field]
            type(str): user or item

        Returns:
            torch.Tensor: token fields embedding, [batch_size, max_item_length, num_token_field, embed_dim]

        """
        if token_fields is None:
            return None
        if type == 'item':
            embedding_shape = token_fields.shape + (-1,)
            token_fields = token_fields.reshape(-1, token_fields.shape[-1])
            token_embedding = self.token_embedding_table[type](token_fields)
            token_embedding = token_embedding.view(embedding_shape)
        else:
            token_embedding = self.token_embedding_table[type](token_fields)
        return token_embedding

    def embed_float_seq_fields(self, float_seq_fields, type):
        """Embed the float sequence feature columns

        Args:
            float_seq_fields (torch.FloatTensor): The input tensor. shape of [batch_size, seq_len, 2]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of float sequence columns.
        """
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[type][i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=-1, keepdim=True)
            float_seq_embedding = base * embedding_table(index.long())
            mask = mask.unsqueeze(-1).expand_as(float_seq_embedding)
            if self.pooling_mode == 'max':
                masked_float_seq_embedding = float_seq_embedding - (1 - mask) * 1000000000.0
                result = torch.max(masked_float_seq_embedding, dim=-2, keepdim=True)
                result = result.values
            elif self.pooling_mode == 'sum':
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=-2, keepdim=True)
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=-2)
                eps = torch.FloatTensor([1e-08])
                result = torch.div(result, value_cnt + eps)
                result = result.unsqueeze(-2)
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=-2)

    def embed_token_seq_fields(self, token_seq_fields, type):
        """Get the embedding of token_seq fields.

        Args:
            token_seq_fields(torch.Tensor): input, [batch_size, max_item_length, seq_len]`
            type(str): user or item
            mode(str): mean/max/sum

        Returns:
            torch.Tensor: result [batch_size, max_item_length, num_token_seq_field, embed_dim]

        """
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[type][i]
            mask = token_seq_field != 0
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=-1, keepdim=True)
            token_seq_embedding = embedding_table(token_seq_field)
            mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
            if self.pooling_mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1000000000.0
                result = torch.max(masked_token_seq_embedding, dim=-2, keepdim=True)
                result = result.values
            elif self.pooling_mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=-2, keepdim=True)
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=-2)
                eps = torch.FloatTensor([1e-08])
                result = torch.div(result, value_cnt + eps)
                result = result.unsqueeze(-2)
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=-2)

    def embed_input_fields(self, user_idx, item_idx):
        """Get the embedding of user_idx and item_idx

        Args:
            user_idx(torch.Tensor): interaction['user_id']
            item_idx(torch.Tensor): interaction['item_id_list']

        Returns:
            dict: embedding of user feature and item feature

        """
        user_item_feat = {'user': self.user_feat, 'item': self.item_feat}
        user_item_idx = {'user': user_idx, 'item': item_idx}
        float_fields_embedding = {}
        float_seq_fields_embedding = {}
        token_fields_embedding = {}
        token_seq_fields_embedding = {}
        sparse_embedding = {}
        dense_embedding = {}
        for type in self.types:
            float_fields = []
            for field_name in self.float_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_fields.append(feature if len(feature.shape) == 3 + (type == 'item') else feature.unsqueeze(-2))
            if len(float_fields) > 0:
                float_fields = torch.cat(float_fields, dim=-1)
            else:
                float_fields = None
            float_fields_embedding[type] = self.embed_float_fields(float_fields, type)
            float_seq_fields = []
            for field_name in self.float_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_seq_fields.append(feature)
            float_seq_fields_embedding[type] = self.embed_float_seq_fields(float_seq_fields, type)
            if float_fields_embedding[type] is None:
                dense_embedding[type] = float_seq_fields_embedding[type]
            elif float_seq_fields_embedding[type] is None:
                dense_embedding[type] = float_fields_embedding[type]
            else:
                dense_embedding[type] = torch.cat([float_fields_embedding[type], float_seq_fields_embedding[type]], dim=-2)
            token_fields = []
            for field_name in self.token_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_fields.append(feature.unsqueeze(-1))
            if len(token_fields) > 0:
                token_fields = torch.cat(token_fields, dim=-1)
            else:
                token_fields = None
            token_fields_embedding[type] = self.embed_token_fields(token_fields, type)
            token_seq_fields = []
            for field_name in self.token_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_seq_fields.append(feature)
            token_seq_fields_embedding[type] = self.embed_token_seq_fields(token_seq_fields, type)
            if token_fields_embedding[type] is None:
                sparse_embedding[type] = token_seq_fields_embedding[type]
            elif token_seq_fields_embedding[type] is None:
                sparse_embedding[type] = token_fields_embedding[type]
            else:
                sparse_embedding[type] = torch.cat([token_fields_embedding[type], token_seq_fields_embedding[type]], dim=-2)
        return sparse_embedding, dense_embedding

    def forward(self, user_idx, item_idx):
        return self.embed_input_fields(user_idx, item_idx)


class ContextSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For Deep Interest Network, return all features (including user features and item features) embedding matrices."""

    def __init__(self, dataset, embedding_size, pooling_mode, device):
        super(ContextSeqEmbLayer, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = self.dataset.get_user_feature()
        self.item_feat = self.dataset.get_item_feature()
        self.field_names = {'user': list(self.user_feat.interaction.keys()), 'item': list(self.item_feat.interaction.keys())}
        self.types = ['user', 'item']
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ['mean', 'max', 'sum']
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class FeatureSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For feature-rich sequential recommenders, return item features embedding matrices according to
    selected features."""

    def __init__(self, dataset, embedding_size, selected_features, pooling_mode, device):
        super(FeatureSeqEmbLayer, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = None
        self.item_feat = self.dataset.get_item_feature()
        self.field_names = {'item': selected_features}
        self.types = ['item']
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ['mean', 'max', 'sum']
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class TransNet(nn.Module):

    def __init__(self, config, dataset):
        super().__init__()
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.position_embedding = nn.Embedding(dataset.field2seqlen[config['ITEM_ID_FIELD'] + config['LIST_SUFFIX']], self.hidden_size)
        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads, hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob, attn_dropout_prob=self.attn_dropout_prob, hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)
        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        alpha = self.fn(output)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9000000000000000.0)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class AGRUCell(nn.Module):
    'Attention based GRU (AGRU). AGRU uses the attention score to replace the update gate of GRU, and changes the\n    hidden state directly.\n\n    Formally:\n        ..math: {h}_{t}^{\\prime}=\\left(1-a_{t}\right) * {h}_{t-1}^{\\prime}+a_{t} * \tilde{{h}}_{t}^{\\prime}\n\n        :math:`{h}_{t}^{\\prime}`, :math:`h_{t-1}^{\\prime}`, :math:`{h}_{t-1}^{\\prime}`,\n        :math: `\tilde{{h}}_{t}^{\\prime}` are the hidden state of AGRU\n\n    '

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        if self.bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hidden_output, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_output, self.weight_hh, self.bias_hh)
        i_r, i_u, i_h = gi.chunk(3, 1)
        h_r, h_u, h_h = gh.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        new_state = torch.tanh(i_h + reset_gate * h_h)
        att_score = att_score.view(-1, 1)
        hy = (1 - att_score) * hidden_output + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    ' Effect of GRU with attentional update gate (AUGRU). AUGRU combines attention mechanism and GRU seamlessly.\n\n    Formally:\n        ..math: \tilde{{u}}_{t}^{\\prime}=a_{t} * {u}_{t}^{\\prime} \\\n                {h}_{t}^{\\prime}=\\left(1-\tilde{{u}}_{t}^{\\prime}\right) \\circ {h}_{t-1}^{\\prime}+\tilde{{u}}_{t}^{\\prime} \\circ \tilde{{h}}_{t}^{\\prime}\n\n    '

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hidden_output, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_output, self.weight_hh, self.bias_hh)
        i_r, i_u, i_h = gi.chunk(3, 1)
        h_r, h_u, h_h = gh.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_u + h_u)
        new_state = torch.tanh(i_h + reset_gate * h_h)
        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1 - update_gate) * hidden_output + update_gate * new_state
        return hy


class DynamicRNN(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, gru='AGRU'):
        super(DynamicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if gru == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, input, att_scores=None, hidden_output=None):
        if not isinstance(input, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError('DynamicRNN only supports packed input and att_scores')
        input, batch_sizes, sorted_indices, unsorted_indices = input
        att_scores = att_scores.data
        max_batch_size = int(batch_sizes[0])
        if hidden_output is None:
            hidden_output = torch.zeros(max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        outputs = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(input[begin:begin + batch], hidden_output[0:batch], att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hidden_output = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)


class InterestEvolvingLayer(nn.Module):
    """As the joint influence from external environment and internal cognition, different kinds of user interests are
    evolving over time. Interest Evolving Layer can capture interest evolving process that is relative to the target
    item.
    """

    def __init__(self, mask_mat, input_size, rnn_hidden_size, att_hidden_size=(80, 40), activation='sigmoid', softmax_stag=True, gru='GRU'):
        super(InterestEvolvingLayer, self).__init__()
        self.mask_mat = mask_mat
        self.gru = gru
        if gru == 'GRU':
            self.attention_layer = SequenceAttLayer(mask_mat, att_hidden_size, activation, softmax_stag, False)
            self.dynamic_rnn = nn.GRU(input_size=input_size, hidden_size=rnn_hidden_size, batch_first=True)
        elif gru == 'AIGRU':
            self.attention_layer = SequenceAttLayer(mask_mat, att_hidden_size, activation, softmax_stag, True)
            self.dynamic_rnn = nn.GRU(input_size=input_size, hidden_size=rnn_hidden_size, batch_first=True)
        elif gru == 'AGRU' or gru == 'AUGRU':
            self.attention_layer = SequenceAttLayer(mask_mat, att_hidden_size, activation, softmax_stag, True)
            self.dynamic_rnn = DynamicRNN(input_size=input_size, hidden_size=rnn_hidden_size, gru=gru)

    def final_output(self, outputs, keys_length):
        """get the last effective value in the interest evolution sequence
        Args:
            outputs (torch.Tensor): the output of `DynamicRNN` after `pad_packed_sequence`
            keys_length (torch.Tensor): the true length of the user history sequence

        Returns:
            torch.Tensor: The user's CTR for the next item
        """
        batch_size, hist_len, _ = outputs.shape
        mask = torch.arange(hist_len, device=keys_length.device).repeat(batch_size, 1) == keys_length.view(-1, 1) - 1
        return outputs[mask]

    def forward(self, queries, keys, keys_length):
        hist_len = keys.shape[1]
        keys_length_cpu = keys_length.cpu()
        if self.gru == 'GRU':
            packed_keys = pack_padded_sequence(input=keys, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False)
            packed_rnn_outputs, _ = self.dynamic_rnn(packed_keys)
            rnn_outputs, _ = pad_packed_sequence(packed_rnn_outputs, batch_first=True, padding_value=0.0, total_length=hist_len)
            att_outputs = self.attention_layer(queries, rnn_outputs, keys_length)
            outputs = att_outputs.squeeze(1)
        elif self.gru == 'AIGRU':
            att_outputs = self.attention_layer(queries, keys, keys_length)
            interest = keys * att_outputs.transpose(1, 2)
            packed_rnn_outputs = pack_padded_sequence(interest, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False)
            _, outputs = self.dynamic_rnn(packed_rnn_outputs)
            outputs = outputs.squeeze(0)
        elif self.gru == 'AGRU' or self.gru == 'AUGRU':
            att_outputs = self.attention_layer(queries, keys, keys_length).squeeze(1)
            packed_rnn_outputs = pack_padded_sequence(keys, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False)
            packed_att_outputs = pack_padded_sequence(att_outputs, lengths=keys_length_cpu, batch_first=True, enforce_sorted=False)
            outputs = self.dynamic_rnn(packed_rnn_outputs, packed_att_outputs)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=hist_len)
            outputs = self.final_output(outputs, keys_length)
        return outputs


class InterestExtractorNetwork(nn.Module):
    """In e-commerce system, user behavior is the carrier of latent interest, and interest will change after
    user takes one behavior. At the interest extractor layer, DIEN extracts series of interest states from
    sequential user behaviors.
    """

    def __init__(self, input_size, hidden_size, mlp_size):
        super(InterestExtractorNetwork, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.auxiliary_net = MLPLayers(layers=mlp_size, activation='none')

    def forward(self, keys, keys_length, neg_keys=None):
        batch_size, hist_len, embedding_size = keys.shape
        packed_keys = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_rnn_outputs, _ = self.gru(packed_keys)
        rnn_outputs, _ = pad_packed_sequence(packed_rnn_outputs, batch_first=True, padding_value=0, total_length=hist_len)
        aux_loss = self.auxiliary_loss(rnn_outputs[:, :-1, :], keys[:, 1:, :], neg_keys[:, 1:, :], keys_length - 1)
        return rnn_outputs, aux_loss

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, keys_length):
        """Computes the auxiliary loss

        Formally:
        ..math: L_{a u x}= \\frac{1}{N}(\\sum_{i=1}^{N} \\sum_{t} \\log \\sigma(\\mathbf{h}_{t}^{i}, \\mathbf{e}_{b}^{i}[t+1])
                + \\log (1-\\sigma(\\mathbf{h}_{t}^{i}, \\hat{\\mathbf{e}}_{b}^{i}[t+1])))

        Args:
            h_states (torch.Tensor): The output of GRUs' hidden layer, [batch_size, history_length - 1, embedding,size].
            click_seq (torch.Tensor): The sequence that users consumed, [batch_size, history_length - 1, embedding,size].
            noclick_seq (torch.Tensor): The sequence that users did not consume, [batch_size, history_length - 1, embedding_size].

         Returns:
            torch.Tensor: auxiliary loss

        """
        batch_size, hist_length, embedding_size = h_states.shape
        click_input = torch.cat([h_states, click_seq], dim=-1)
        noclick_input = torch.cat([h_states, noclick_seq], dim=-1)
        mask = (torch.arange(hist_length, device=h_states.device).repeat(batch_size, 1) < keys_length.view(-1, 1)).float()
        click_prop = self.auxiliary_net(click_input.view(batch_size * hist_length, -1)).view(batch_size, hist_length)[mask > 0].view(-1, 1)
        click_target = torch.ones(click_prop.shape, device=click_input.device)
        noclick_prop = self.auxiliary_net(noclick_input.view(batch_size * hist_length, -1)).view(batch_size, hist_length)[mask > 0].view(-1, 1)
        noclick_target = torch.zeros(noclick_prop.shape, device=noclick_input.device)
        loss = F.binary_cross_entropy_with_logits(torch.cat([click_prop, noclick_prop], dim=0), torch.cat([click_target, noclick_target], dim=0))
        return loss


class HybridAttention(nn.Module):
    """
    Hybrid Attention layer: combine time domain self-attention layer and frequency domain attention layer.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head Hybrid Attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head Hybrid Attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, i, config):
        super(HybridAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (hidden_size, n_heads))
        self.factor = config['topk_factor']
        self.scale = None
        self.mask_flag = True
        self.output_attention = False
        self.dropout = nn.Dropout(0.1)
        self.config = config
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.filter_mixer = None
        self.global_ratio = config['global_ratio']
        self.n_layers = config['n_layers']
        if self.global_ratio > 1 / self.n_layers:
            None
            self.filter_mixer = 'G'
        else:
            None
            self.filter_mixer = 'L'
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.dual_domain = config['dual_domain']
        self.slide_step = (self.max_item_list_length // 2 + 1) * (1 - self.global_ratio) // (self.n_layers - 1)
        self.local_ratio = 1 / self.n_layers
        self.filter_size = self.local_ratio * (self.max_item_list_length // 2 + 1)
        if self.filter_mixer == 'G':
            self.w = self.global_ratio
            self.s = self.slide_step
        if self.filter_mixer == 'L':
            self.w = self.local_ratio
            self.s = self.filter_size
        self.left = int((self.max_item_list_length // 2 + 1) * (1 - self.w) - i * self.s)
        self.right = int(self.max_item_list_length // 2 + 1 - i * self.s)
        self.q_index = list(range(self.left, self.right))
        self.k_index = list(range(self.left, self.right))
        self.v_index = list(range(self.left, self.right))
        self.std = config['std']
        if self.std:
            self.time_q_index = self.q_index
            self.time_k_index = self.k_index
            self.time_v_index = self.v_index
        else:
            self.time_q_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_k_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_v_index = list(range(self.max_item_list_length // 2 + 1))
        None
        None
        None
        if self.config['dual_domain']:
            self.spatial_ratio = self.config['spatial_ratio']

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        return delays_agg

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query_layer(input_tensor)
        mixed_key_layer = self.key_layer(input_tensor)
        mixed_value_layer = self.value_layer(input_tensor)
        queries = self.transpose_for_scores(mixed_query_layer)
        keys = self.transpose_for_scores(mixed_key_layer)
        values = self.transpose_for_scores(mixed_value_layer)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :L - S, :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        q_fft_box = torch.zeros(B, H, E, len(self.q_index), device=q_fft.device, dtype=torch.cfloat)
        q_fft_box = q_fft[:, :, :, self.q_index]
        k_fft_box = torch.zeros(B, H, E, len(self.k_index), device=q_fft.device, dtype=torch.cfloat)
        k_fft_box = k_fft[:, :, :, self.q_index]
        res = q_fft_box * torch.conj(k_fft_box)
        if self.config['use_filter']:
            weight = torch.view_as_complex(self.complex_weight)
            res = res * weight
        box_res = torch.zeros(B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat)
        box_res[:, :, :, self.q_index] = res
        corr = torch.fft.irfft(box_res, dim=-1)
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        new_context_layer_shape = V.size()[:-2] + (self.all_head_size,)
        context_layer = V.view(*new_context_layer_shape)
        if self.dual_domain:
            q_fft_box = torch.zeros(B, H, E, len(self.time_q_index), device=q_fft.device, dtype=torch.cfloat)
            q_fft_box = q_fft[:, :, :, self.time_q_index]
            spatial_q = torch.zeros(B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat)
            spatial_q[:, :, :, self.time_q_index] = q_fft_box
            k_fft_box = torch.zeros(B, H, E, len(self.time_k_index), device=q_fft.device, dtype=torch.cfloat)
            k_fft_box = k_fft[:, :, :, self.time_k_index]
            spatial_k = torch.zeros(B, H, E, L // 2 + 1, device=k_fft.device, dtype=torch.cfloat)
            spatial_k[:, :, :, self.time_k_index] = k_fft_box
            v_fft = torch.fft.rfft(values.permute(0, 2, 3, 1).contiguous(), dim=-1)
            v_fft_box = torch.zeros(B, H, E, len(self.time_v_index), device=v_fft.device, dtype=torch.cfloat)
            v_fft_box = v_fft[:, :, :, self.time_v_index]
            spatial_v = torch.zeros(B, H, E, L // 2 + 1, device=v_fft.device, dtype=torch.cfloat)
            spatial_v[:, :, :, self.time_v_index] = v_fft_box
            queries = torch.fft.irfft(spatial_q, dim=-1)
            keys = torch.fft.irfft(spatial_k, dim=-1)
            values = torch.fft.irfft(spatial_v, dim=-1)
            queries = queries.permute(0, 1, 3, 2)
            keys = keys.permute(0, 1, 3, 2)
            values = values.permute(0, 1, 3, 2)
            attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)
            qkv = torch.matmul(attention_probs, values)
            context_layer_spatial = qkv.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer_spatial.size()[:-2] + (self.all_head_size,)
            context_layer_spatial = context_layer_spatial.view(*new_context_layer_shape)
            context_layer = (1 - self.spatial_ratio) * context_layer + self.spatial_ratio * context_layer_spatial
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FEABlock(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, n, config):
        super(FEABlock, self).__init__()
        self.hybrid_attention = HybridAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, n, config)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.hybrid_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class FEAEncoder(nn.Module):
    """One TransformerEncoder consists of several TransformerLayers.

    - n_layers(num): num of transformer layers in transformer encoder. Default: 2
    - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
    - hidden_size(num): the input and output hidden size. Default: 64
    - inner_size(num): the dimensionality in feed-forward layer. Default: 256
    - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
    - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
    - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                  candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
    - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(self, n_layers=2, n_heads=2, hidden_size=64, inner_size=256, hidden_dropout_prob=0.5, attn_dropout_prob=0.5, hidden_act='gelu', layer_norm_eps=1e-12, config=None):
        super(FEAEncoder, self).__init__()
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        for n in range(self.n_layers):
            self.layer_ramp = FEABlock(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, n, config)
            self.layer.append(self.layer_ramp)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def _convert_to_tensor(data):
    """This function can convert common data types (list, pandas.Series, numpy.ndarray, torch.Tensor) into torch.Tensor.

    Args:
        data (list, pandas.Series, numpy.ndarray, torch.Tensor): Origin data.

    Returns:
        torch.Tensor: Converted tensor from `data`.
    """
    elem = data[0]
    if isinstance(elem, (float, int, np.float, np.int64)):
        new_data = torch.as_tensor(data)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        seq_data = [torch.as_tensor(d) for d in data]
        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True)
    else:
        raise ValueError(f'[{type(elem)}] is not supported!')
    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    return new_data


class Interaction(object):
    """The basic class representing a batch of interaction records.

    Note:
        While training, there is no strict rules for data in one Interaction object.

        While testing, it should be guaranteed that all interaction records of one single
        user will not appear in different Interaction object, and records of the same user
        should be continuous. Meanwhile, the positive cases of one user always need to occur
        **earlier** than this user's negative cases.

        A correct example:
            =======     =======     =======
            user_id     item_id     label
            =======     =======     =======
            1           2           1
            1           6           1
            1           3           1
            1           1           0
            2           3           1
            ...         ...         ...
            =======     =======     =======

        Some wrong examples for Interaction objects used in testing:

        1.
            =======     =======     =======     ============
            user_id     item_id     label
            =======     =======     =======     ============
            1           2           1
            1           6           0           # positive cases of one user always need to

                                                occur earlier than this user's negative cases
            1           3           1
            1           1           0
            2           3           1
            ...         ...         ...
            =======     =======     =======     ============

        2.
            =======     =======     =======     ========
            user_id     item_id     label
            =======     =======     =======     ========
            1           2           1
            1           6           1
            1           3           1
            2           3           1           # records of the same user should be continuous.
            1           1           0
            ...         ...         ...
            =======     =======     =======     ========

    Attributes:
        interaction (dict or pandas.DataFrame): keys are meaningful str (also can be called field name),
            and values are Torch Tensor of numpy Array with shape (batch_size, \\*).
    """

    def __init__(self, interaction):
        self.interaction = dict()
        if isinstance(interaction, dict):
            for key, value in interaction.items():
                if isinstance(value, (list, np.ndarray)):
                    self.interaction[key] = _convert_to_tensor(value)
                elif isinstance(value, torch.Tensor):
                    self.interaction[key] = value
                else:
                    raise ValueError(f'The type of {key}[{type(value)}] is not supported!')
        elif isinstance(interaction, pd.DataFrame):
            for key in interaction:
                value = interaction[key].values
                self.interaction[key] = _convert_to_tensor(value)
        else:
            raise ValueError(f'[{type(interaction)}] is not supported for initialize `Interaction`!')
        self.length = -1
        for k in self.interaction:
            self.length = max(self.length, self.interaction[k].unsqueeze(-1).shape[0])

    def __iter__(self):
        return self.interaction.__iter__()

    def __getattr__(self, item):
        if 'interaction' not in self.__dict__:
            raise AttributeError(f"'Interaction' object has no attribute 'interaction'")
        if item in self.interaction:
            return self.interaction[item]
        raise AttributeError(f"'Interaction' object has no attribute '{item}'")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        if isinstance(index, (np.ndarray, torch.Tensor)):
            index = index.tolist()
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k][index]
        return Interaction(ret)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError(f'{type(key)} object does not support item assigment')
        self.interaction[key] = value

    def __delitem__(self, key):
        if key not in self.interaction:
            raise KeyError(f'{type(key)} object does not in this interaction')
        del self.interaction[key]

    def __contains__(self, item):
        return item in self.interaction

    def __len__(self):
        return self.length

    def __str__(self):
        info = [f'The batch_size of interaction: {self.length}']
        for k in self.interaction:
            inter = self.interaction[k]
            temp_str = f'    {k}, {inter.shape}, {inter.device.type}, {inter.dtype}'
            info.append(temp_str)
        info.append('\n')
        return '\n'.join(info)

    def __repr__(self):
        return self.__str__()

    @property
    def columns(self):
        """
        Returns:
            list of str: The columns of interaction.
        """
        return list(self.interaction.keys())

    def to(self, device, selected_field=None):
        """Transfer Tensors in this Interaction object to the specified device.

        Args:
            device (torch.device): target device.
            selected_field (str or iterable object, optional): if specified, only Tensors
            with keys in selected_field will be sent to device.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to
            the specified device.
        """
        ret = {}
        if isinstance(selected_field, str):
            selected_field = [selected_field]
        if selected_field is not None:
            selected_field = set(selected_field)
            for k in self.interaction:
                if k in selected_field:
                    ret[k] = self.interaction[k]
                else:
                    ret[k] = self.interaction[k]
        else:
            for k in self.interaction:
                ret[k] = self.interaction[k]
        return Interaction(ret)

    def cpu(self):
        """Transfer Tensors in this Interaction object to cpu.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to cpu.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].cpu()
        return Interaction(ret)

    def numpy(self):
        """Transfer Tensors to numpy arrays.

        Returns:
            dict: keys the same as Interaction object, are values are corresponding numpy
            arrays transformed from Tensor.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].numpy()
        return ret

    def repeat(self, sizes):
        """Repeats each tensor along the batch dim.

        Args:
            sizes (int): repeat times.

        Example:
            >>> a = Interaction({'k': torch.zeros(4)})
            >>> a.repeat(3)
            The batch_size of interaction: 12
                k, torch.Size([12]), cpu

            >>> a = Interaction({'k': torch.zeros(4, 7)})
            >>> a.repeat(3)
            The batch_size of interaction: 12
                k, torch.Size([12, 7]), cpu

        Returns:
            a copyed Interaction object with repeated Tensors.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat([sizes] + [1] * (len(self.interaction[k].shape) - 1))
        return Interaction(ret)

    def repeat_interleave(self, repeats, dim=0):
        """Similar to repeat_interleave of PyTorch.

        Details can be found in:

            https://pytorch.org/docs/stable/tensors.html?highlight=repeat#torch.Tensor.repeat_interleave

        Note:
            ``torch.repeat_interleave()`` is supported in PyTorch >= 1.2.0.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat_interleave(repeats, dim=dim)
        return Interaction(ret)

    def update(self, new_inter):
        """Similar to ``dict.update()``

        Args:
            new_inter (Interaction): current interaction will be updated by new_inter.
        """
        for k in new_inter.interaction:
            self.interaction[k] = new_inter.interaction[k]

    def drop(self, column):
        """Drop column in interaction.

        Args:
            column (str): the column to be dropped.
        """
        if column not in self.interaction:
            raise ValueError(f'Column [{column}] is not in [{self}].')
        del self.interaction[column]

    def _reindex(self, index):
        """Reset the index of interaction inplace.

        Args:
            index: the new index of current interaction.
        """
        for k in self.interaction:
            self.interaction[k] = self.interaction[k][index]

    def shuffle(self):
        """Shuffle current interaction inplace."""
        index = torch.randperm(self.length)
        self._reindex(index)

    def sort(self, by, ascending=True):
        """Sort the current interaction inplace.

        Args:
            by (str or list of str): Field that as the key in the sorting process.
            ascending (bool or list of bool, optional): Results are ascending if ``True``, otherwise descending.
                Defaults to ``True``
        """
        if isinstance(by, str):
            if by not in self.interaction:
                raise ValueError(f'[{by}] is not exist in interaction [{self}].')
            by = [by]
        elif isinstance(by, (list, tuple)):
            for b in by:
                if b not in self.interaction:
                    raise ValueError(f'[{b}] is not exist in interaction [{self}].')
        else:
            raise TypeError(f'Wrong type of by [{by}].')
        if isinstance(ascending, bool):
            ascending = [ascending]
        elif isinstance(ascending, (list, tuple)):
            for a in ascending:
                if not isinstance(a, bool):
                    raise TypeError(f'Wrong type of ascending [{ascending}].')
        else:
            raise TypeError(f'Wrong type of ascending [{ascending}].')
        if len(by) != len(ascending):
            if len(ascending) == 1:
                ascending = ascending * len(by)
            else:
                raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')
        for b, a in zip(by[::-1], ascending[::-1]):
            if len(self.interaction[b].shape) == 1:
                key = self.interaction[b]
            else:
                key = self.interaction[b][..., 0]
            index = np.argsort(key, kind='stable')
            if not a:
                index = torch.tensor(np.array(index)[::-1])
            self._reindex(index)

    def add_prefix(self, prefix):
        """Add prefix to current interaction's columns.

        Args:
            prefix (str): The prefix to be added.
        """
        self.interaction = {(prefix + key): value for key, value in self.interaction.items()}


class GNN(nn.Module):
    """Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))
        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

    def GNNCell(self, A, hidden):
        """Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """
        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden)) + self.b_ioh
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ResidualBlock_b(nn.Module):
    """
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-08)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-08)
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):
        x_pad = self.conv_pad(x, self.dilation)
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        """Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


class ResidualBlock_a(nn.Module):
    """
    Residual block (a) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_a, self).__init__()
        half_channel = out_channel // 2
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-08)
        self.conv1 = nn.Conv2d(in_channel, half_channel, kernel_size=(1, 1), padding=0)
        self.ln2 = nn.LayerNorm(half_channel, eps=1e-08)
        self.conv2 = nn.Conv2d(half_channel, half_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln3 = nn.LayerNorm(half_channel, eps=1e-08)
        self.conv3 = nn.Conv2d(half_channel, out_channel, kernel_size=(1, 1), padding=0)
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):
        out = F.relu(self.ln1(x))
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.conv1(out).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out))
        out2 = self.conv_pad(out2, self.dilation)
        out2 = self.conv2(out2).squeeze(2).permute(0, 2, 1)
        out3 = F.relu(self.ln3(out2))
        out3 = out3.permute(0, 2, 1).unsqueeze(2)
        out3 = self.conv3(out3).squeeze(2).permute(0, 2, 1)
        return out3 + x

    def conv_pad(self, x, dilation):
        """Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


class Explore_Recommendation_Decoder(nn.Module):

    def __init__(self, hidden_size, seq_len, num_item, device, dropout_prob):
        super(Explore_Recommendation_Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_item = num_item
        self.device = device
        self.We = nn.Linear(hidden_size, hidden_size)
        self.Ue = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.Ve = nn.Linear(hidden_size, 1)
        self.matrix_for_explore = nn.Linear(2 * self.hidden_size, self.num_item, bias=False)

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        """
        calculate the force of explore
        """
        all_memory_values, last_memory_values = all_memory, last_memory
        all_memory = self.dropout(self.Ue(all_memory))
        last_memory = self.dropout(self.We(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)
        output_ee = self.tanh(all_memory + last_memory)
        output_ee = self.Ve(output_ee).squeeze(-1)
        if mask is not None:
            output_ee.masked_fill_(mask, -1000000000.0)
        output_ee = output_ee.unsqueeze(-1)
        alpha_e = nn.Softmax(dim=1)(output_ee)
        alpha_e = alpha_e.repeat(1, 1, self.hidden_size)
        output_e = (alpha_e * all_memory_values).sum(dim=1)
        output_e = torch.cat([output_e, last_memory_values], dim=1)
        output_e = self.dropout(self.matrix_for_explore(output_e))
        item_seq_first = item_seq[:, 0].unsqueeze(1).expand_as(item_seq)
        item_seq_first = item_seq_first.masked_fill(item_seq > 0, 0)
        item_seq_first.requires_grad_(False)
        output_e.scatter_add_(1, item_seq + item_seq_first, float('-inf') * torch.ones_like(item_seq))
        explore_recommendation_decoder = nn.Softmax(1)(output_e)
        return explore_recommendation_decoder


class Repeat_Explore_Mechanism(nn.Module):

    def __init__(self, device, hidden_size, seq_len, dropout_prob):
        super(Repeat_Explore_Mechanism, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.seq_len = seq_len
        self.Wre = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ure = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vre = nn.Linear(hidden_size, 1, bias=False)
        self.Wcre = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, all_memory, last_memory):
        """
        calculate the probability of Repeat and explore
        """
        all_memory_values = all_memory
        all_memory = self.dropout(self.Ure(all_memory))
        last_memory = self.dropout(self.Wre(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)
        output_ere = self.tanh(all_memory + last_memory)
        output_ere = self.Vre(output_ere)
        alpha_are = nn.Softmax(dim=1)(output_ere)
        alpha_are = alpha_are.repeat(1, 1, self.hidden_size)
        output_cre = alpha_are * all_memory_values
        output_cre = output_cre.sum(dim=1)
        output_cre = self.Wcre(output_cre)
        repeat_explore_mechanism = nn.Softmax(dim=-1)(output_cre)
        return repeat_explore_mechanism


class Repeat_Recommendation_Decoder(nn.Module):

    def __init__(self, device, hidden_size, seq_len, num_item, dropout_prob):
        super(Repeat_Recommendation_Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.device = device
        self.seq_len = seq_len
        self.num_item = num_item
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.Vr = nn.Linear(hidden_size, 1)

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        """
        calculate the the force of repeat
        """
        all_memory = self.dropout(self.Ur(all_memory))
        last_memory = self.dropout(self.Wr(last_memory))
        last_memory = last_memory.unsqueeze(1)
        last_memory = last_memory.repeat(1, self.seq_len, 1)
        output_er = self.tanh(last_memory + all_memory)
        output_er = self.Vr(output_er).squeeze(2)
        if mask is not None:
            output_er.masked_fill_(mask, -1000000000.0)
        output_er = nn.Softmax(dim=-1)(output_er)
        batch_size, b_len = item_seq.size()
        repeat_recommendation_decoder = torch.zeros([batch_size, self.num_item], device=self.device)
        repeat_recommendation_decoder.scatter_add_(1, item_seq, output_er)
        return repeat_recommendation_decoder


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([16, 4]), torch.rand([16, 4]), torch.rand([4, 4])], {}),
     True),
    (AUGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([64, 4]), torch.rand([64, 4]), torch.rand([16, 4])], {}),
     True),
    (AttLayer,
     lambda: ([], {'in_dim': 4, 'att_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BPRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BaseFactorizationMachine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BiGNNLayer,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (CINComp,
     lambda: ([], {'indim': 4, 'outdim': 4, 'config': _mock_config(feature_num=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNCFBPRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossNet,
     lambda: ([], {'config': _mock_config(depth=1, embedding_size=4, feature_num=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Dice,
     lambda: ([], {'emb_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Explore_Recommendation_Decoder,
     lambda: ([], {'hidden_size': 4, 'seq_len': 4, 'num_item': 4, 'device': 0, 'dropout_prob': 0.5}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64)], {}),
     False),
    (GraphLayer,
     lambda: ([], {'num_fields': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (InnerProductLayer,
     lambda: ([], {'num_feature_field': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InnerProductLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ItemToInterestAggregation,
     lambda: ([], {'seq_len': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MLPLayers,
     lambda: ([], {'layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'n_heads': 4, 'hidden_size': 4, 'hidden_dropout_prob': 0.5, 'attn_dropout_prob': 0.5, 'layer_norm_eps': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (NegSamplingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (OuterProductLayer,
     lambda: ([], {'num_feature_field': 4, 'embedding_size': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RegLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Repeat_Explore_Mechanism,
     lambda: ([], {'device': 0, 'hidden_size': 4, 'seq_len': 4, 'dropout_prob': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (SparseDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VanillaAttention,
     lambda: ([], {'hidden_dim': 4, 'attn_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_RUCAIBox_RecBole(_paritybench_base):
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

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

