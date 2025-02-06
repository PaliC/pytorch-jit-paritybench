import sys
_module = sys.modules[__name__]
del sys
run_multi_task_benchmark_example = _module
run_multi_task_example = _module
inference_example = _module
run_ranking_benchmark_example = _module
run_ranking_example = _module
run_ranking_wandb_example = _module
run_set_pretrained_emb_example = _module
custom_model = _module
run_sequence_example = _module
run_sequence_example_v2 = _module
rec_pangu = _module
benchmark_trainer = _module
dataset = _module
base_dataset = _module
graph_dataset = _module
multi_task_dataset = _module
process_data = _module
sequence_dataset = _module
model_pipeline = _module
models = _module
base_model = _module
ngcf = _module
layers = _module
activation = _module
attention = _module
conv = _module
deep = _module
embedding = _module
graph = _module
interaction = _module
multi_interest = _module
sequence = _module
shallow = _module
trainformer = _module
multi_task = _module
aitm = _module
essm = _module
mlmmoe = _module
mmoe = _module
omoe = _module
sharebottom = _module
ranking = _module
afm = _module
afn = _module
aoanet = _module
autoint = _module
ccpm = _module
dcn = _module
deepfm = _module
fibinet = _module
fm = _module
lr = _module
masknet = _module
nfm = _module
wdl = _module
xdeepfm = _module
clrec = _module
cmi = _module
comirec = _module
contrarec = _module
gcsan = _module
gru4rec = _module
iocrec = _module
mind = _module
narm = _module
nextitnet = _module
niser = _module
re4 = _module
sasrec = _module
sine = _module
srgnn = _module
stamp = _module
yotubednn = _module
utils = _module
serving = _module
ranking_server = _module
trainer = _module
check_version = _module
evaluate = _module
gpu_utils = _module
json_utils = _module
setup = _module

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


import pandas as pd


import numpy as np


from torch import nn


import torch.nn.functional as F


from typing import Dict


from typing import List


from typing import Optional


from torch.utils.data import DataLoader


import time


from torch.utils.data import Dataset


from collections import defaultdict


import random


import torch.utils.data as D


from sklearn.metrics import roc_auc_score


from sklearn.metrics import log_loss


from torch.nn.init import xavier_normal_


from torch.nn.init import constant_


from typing import Union


import torch.nn as nn


from itertools import product


from itertools import combinations


import copy


import math


import torch.nn.functional as fn


from typing import Tuple


from torch.optim import lr_scheduler


from sklearn.preprocessing import normalize


class EmbeddingLayer(nn.Module):

    def __init__(self, enc_dict: 'Dict[str, Dict[str, Union[int, str]]]', embedding_dim: 'int') ->None:
        """
        Initialize EmbeddingLayer instance.
        Args:
            enc_dict: Encoding dictionary containing vocabulary size for each categorical feature
            embedding_dim: Number of dimensions in the embedding space
        """
        super().__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()
        self.emb_feature = []
        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col: nn.Embedding(num_embeddings=self.enc_dict[col]['vocab_size'] + 1, embedding_dim=self.embedding_dim)})

    def set_weights(self, col_name: 'str', embedding_matrix: 'torch.Tensor', trainable: 'Optional[bool]'=True) ->None:
        """
        Set the weights for the embedding layer.
        Args:
            col_name: Column name
            embedding_matrix: Embedding weight tensor for the column name
            trainable: Boolean indicating if the embedding layer should be trained
        """
        self.embedding_layer[col_name].weight = nn.Parameter(embedding_matrix)
        if not trainable:
            self.embedding_layer[col_name].weight.requires_grad = False

    def forward(self, X: 'Dict[str, torch.Tensor]', name: 'Optional[str]'=None) ->torch.Tensor:
        """
        Compute the embeddings for a batch of input tensors.
        Args:
            X: Input tensor of shape [batch_size,feature_dim] where feature_dim is the number of features
            name: String indicating the column name
        Returns:
            feature_emb_list: Tensor of shape [batch_size, num_embeddings] containing embeddings for each input feature.
        """
        if name is None:
            feature_emb_list = []
            for col in self.emb_feature:
                inp = X[col].long().view(-1, 1)
                feature_emb_list.append(self.embedding_layer[col](inp))
            return torch.stack(feature_emb_list, dim=1).squeeze(2)
        else:
            if 'seq' in name:
                inp = X[name].long()
                fea = self.embedding_layer[name.replace('_seq', '')](inp)
            else:
                inp = X[name].long().view(-1, 1)
                fea = self.embedding_layer[name](inp)
            return fea


class BaseModel(nn.Module):

    def __init__(self, enc_dict: 'dict', embedding_dim: 'int') ->None:
        """
        A base class for a neural network model.

        Args:
            enc_dict (dict): A dictionary containing the encoding details.
            embedding_dim (int): Dimension of the embedding layer.
        """
        super().__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)

    def _init_weights(self, module: 'nn.Module') ->None:
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): A neural network module.
        """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def reset_parameters(self):
        """
        Initializes the weights of the neural network.

        Args:
            self: The neural network object.

        Returns:
            None
        """
        for weight in self.parameters():
            if len(weight.shape) == 1:
                continue
            else:
                torch.nn.init.kaiming_normal_(weight)

    def set_pretrained_weights(self, col_name: 'str', pretrained_dict: 'dict', trainable: 'bool'=True) ->None:
        """
        Set the pre-trained weights for the model.

        Args:
            col_name (str): Column name for the embedding layer.
            pretrained_dict (dict): A pre-trained embedding dictionary.
            trainable (bool, optional): Training flag. Defaults to True.

        Raises:
            AssertionError: If the column name is not in the encoding dictionary.
                              If the pre-trained embedding dimension is not equal to the model embedding dimension.
        """
        assert col_name in self.enc_dict.keys(), 'Pretrained Embedding Col: {} must be in the {}'.format(col_name, self.enc_dict.keys())
        pretrained_emb_dim = len(list(pretrained_dict.values())[0])
        assert self.embedding_dim == pretrained_emb_dim, 'Pretrained Embedding Dim:{} must be equal to Model Embedding Dim:{}'.format(pretrained_emb_dim, self.embedding_dim)
        pretrained_emb = np.random.rand(self.enc_dict[col_name]['vocab_size'], pretrained_emb_dim)
        for k, v in self.enc_dict[col_name].items():
            if k == 'vocab_size':
                continue
            pretrained_emb[v, :] = pretrained_dict.get(k, np.random.rand(pretrained_emb_dim))
        embeddings = torch.from_numpy(pretrained_emb).float()
        embedding_matrix = torch.nn.Parameter(embeddings)
        self.embedding_layer.set_weights(col_name=col_name, embedding_matrix=embedding_matrix, trainable=trainable)
        logger.info('Successfully Set The Pretrained Embedding Weights for the column:{} With Trainable={}'.format(col_name, trainable))


class SequenceBaseModel(nn.Module):
    """
    Base sequence model for recommendation tasks.

    Attributes:
    enc_dict (dict): A dictionary mapping categorical variable names to their respective encoding dictionaries.
    config (dict): A dictionary containing model hyperparameters such as the embedding size, max sequence length, and device.
    embedding_dim (int): The embedding dimension size.
    max_length (int): The maximum length for input sequences.
    device (str): The device on which the model is run.
    item_emb (nn.Embedding): An embedding layer for item features.
    loss_fun: (nn.CrossEntropyLoss): A loss function used for training.
    """

    def __init__(self, enc_dict: 'dict', config: 'dict'):
        super().__init__()
        self.enc_dict = enc_dict
        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.device = self.config['device']
        self.item_emb = nn.Embedding(self.enc_dict[self.config['item_col']]['vocab_size'], self.embedding_dim, padding_idx=0)
        for col in self.config['cate_cols']:
            setattr(self, f'{col}_emb', nn.Embedding(self.enc_dict[col]['vocab_size'], self.embedding_dim, padding_idx=0))
        self.loss_fun = nn.CrossEntropyLoss()

    def calculate_loss(self, user_emb: 'torch.Tensor', pos_item: 'torch.Tensor') ->torch.Tensor:
        """
        Calculates the loss for the model given a user embedding and positive item.

        Args:
        user_emb (torch.Tensor): A tensor representing the user embedding.
        pos_item (torch.Tensor): A tensor representing the positive item.

        Returns:
        The tensor representing the calculated loss value.
        """
        all_items = self.output_items()
        scores = torch.matmul(user_emb, all_items.transpose(1, 0))
        loss = self.loss_fun(scores, pos_item)
        return loss

    def gather_indexes(self, output: 'torch.Tensor', gather_index: 'torch.Tensor') ->torch.Tensor:
        """
        Gathers the vectors at the specific positions over a minibatch.

        Args:
        output (torch.Tensor): A tensor representing the output vectors.
        gather_index (torch.Tensor): A tensor representing the index vectors.

        Returns:
        The tensor representing the gathered output vectors.
        """
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def output_items(self) ->torch.Tensor:
        """
        Returns the item embedding layer weight.

        Returns:
        The tensor representing the item embedding layer weight.
        """
        return self.item_emb.weight

    def get_attention_mask(self, attention_mask: 'torch.Tensor') ->torch.Tensor:
        """
        Generate left-to-right uni-directional attention mask for multi-head attention.

        Args:
        attention_mask: a tensor used in multi-head attention with shape (batch_size,
        seq_len), containing values of either 0 or 1. 0 indicates padding of a sequence
        and 1 indicates the actual content of the sequence.

        Return:
        extended_attention_mask: a tensor with shape (batch_size, 1, seq_len, seq_len).
        An attention mask tensor with float values of -1e6 added to masked positions
        and 0 to unmasked positions.
        """
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = 1, max_len, max_len
        subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).type_as(attention_mask)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        return extended_attention_mask

    def _init_weights(self, module: 'nn.Module'):
        """
        Initializes the weight value for the given module.

        Args:
        module (nn.Module): The module whose weights need to be initialized.
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)

    def reset_parameters(self):
        """
        Initializes the weights of the neural network.

        Args:
            self: The neural network object.

        Returns:
            None
        """
        for weight in self.parameters():
            if len(weight.shape) == 1:
                continue
            else:
                torch.nn.init.kaiming_normal_(weight)


class GraphBaseModel(nn.Module):

    def __int__(self, num_user, num_item, embedding_dim):
        super(GraphBaseModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_user = num_item
        self.num_item = num_item
        self.user_emb_layer = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_emb_layer = nn.Embedding(self.num_item, self.embedding_dim)

    def reset_parameters(self):
        """
        Initializes the weights of the neural network.

        Args:
            self: The neural network object.

        Returns:
            None
        """
        for weight in self.parameters():
            if len(weight.shape) == 1:
                continue
            else:
                torch.nn.init.kaiming_normal_(weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)
        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss
        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]
        return mf_loss + emb_loss

    def get_ego_embedding(self):
        user_emb = self.user_emb_layer.weight
        item_emb = self.item_emb_layer.weight
        return torch.cat([user_emb, item_emb], 0)


class NGCFLayer(nn.Module):

    def __init__(self, in_size, out_size, dropout):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout
        self.in_size = in_size
        self.out_size = out_size
        self.W1 = nn.Linear(in_size, out_size, bias=False)
        self.W2 = nn.Linear(in_size, out_size, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def message_fun(self, edges):
        edge_feature = self.W1(edges.src['h']) + self.W2(edges.src['h'] * edges.dst['h'])
        edge_feature = edge_feature * (edges.src['norm'] * edges.dst['norm'])
        return {'e': edge_feature}

    def forward(self, g, ego_embedding):
        g.ndata['h'] = ego_embedding
        g.update_all(message_func=self.message_fun, reduce_func=fn.sum('e', 'h_N'))
        g.ndata['h_N'] = g.ndata['h_N'] + self.W1(g.ndata['h'])
        h = self.leaky_relu(g.ndata['h_N'])
        h = self.dropout(h)
        h = F.normalize(h, dim=1, p=2)
        return h


class NGCF(GraphBaseModel):

    def __init__(self, g, num_user, num_item, embedding_dim, hidden_size, dropout=0.1, lmbd=1e-05):
        super(NGCF, self).__init__(num_user, num_item, embedding_dim)
        self.g = g
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lmbd = lmbd
        self.ngcf_layers = nn.ModuleList()
        self.hidden_size = [self.embedding_dim] + self.hidden_size
        for i in range(len(self.hidden_size) - 1):
            self.ngcf_layers.append(NGCFLayer(self.hidden_size[i], self.hidden_size[i + 1], self.dropout))
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        ego_embedding = self.get_ego_embedding()
        user_embeds = []
        item_embeds = []
        user_embeds.append(self.user_emb_layer.weight)
        item_embeds.append(self.item_emb_layer.weight)
        for ngcf_layer in self.ngcf_layers:
            ego_embedding = ngcf_layer(self.g, ego_embedding)
            temp_user_emb, temp_item_emb = torch.split(ego_embedding, [self.num_user, self.num_item])
            user_embeds.append(temp_user_emb)
            item_embeds.append(temp_item_emb)
        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)
        output_dict = dict()
        if is_training:
            u_g_embeddings = user_embd[data['user_id'], :]
            pos_i_g_embeddings = item_embd[data['pos_item_id'], :]
            neg_i_g_embeddings = item_embd[data['neg_item_id'], :]
            loss = self.create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
            output_dict['loss'] = loss
        else:
            output_dict['user_emb'] = user_embd
            output_dict['item_emb'] = item_embd
        return output_dict


class Dice(nn.Module):
    """Implements the Dice activation function.

    Args:
        input_dim (int): dimensionality of the input tensor
        eps (float, optional): term added to the denominator to provide numerical stability (default: 1e-9)
    """

    def __init__(self, input_dim: 'int', eps: 'float'=1e-09):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Performs forward pass of the Dice activation function.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            output (torch.Tensor): output tensor after applying the Dice activation function
        """
        p = torch.sigmoid(self.bn(x))
        output = p * x + (1 - p) * self.alpha * x
        return output


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, dropout_rate=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=None):
        attention = torch.bmm(W_q, W_k.transpose(1, 2))
        if scale:
            attention = attention / scale
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)
        return output, attention


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


class MultiHeadSelfAttention(MultiHeadAttention):

    def forward(self, X):
        output, _ = super(MultiHeadSelfAttention, self).forward(X, X, X)
        return output


class SqueezeExcitationLayer(nn.Module):

    def __init__(self, num_fields, reduction_ratio=3):
        super(SqueezeExcitationLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False), nn.ReLU(), nn.Linear(reduced_size, num_fields, bias=False), nn.ReLU())

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class LayerNorm(nn.Module):
    """
    Layer normalization operation.
    Args:
        channels (int): Number of channels in the input tensor.
        epsilon (float, optional): Small number to avoid numerical instability.
    """

    def __init__(self, channels: 'int', epsilon: 'float'=1e-05) ->None:
        """
        Initialize layer normalization.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones([1, channels, 1], dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros([1, channels, 1], dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Forward pass of layer normalization.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_channels, sequence_length).
        Returns:
            The tensor normalized by layer norm operation.
        """
        var, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.epsilon)
        return x * self.gamma + self.beta


class MaskedConv1d(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', dilation: 'int'=1) ->None:
        """
        This class implements a 1d convolutional neural network.

        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Size of the kernel/window.
        - dilation (int): Controls the spacing between the kernel points. Default is 1.

        Returns:
        - None
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.padding = (kernel_size - 1) * dilation

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Convolves the input tensor using the parameters set during initialization.

        Args:
        - x (torch.Tensor): Input tensor with size (B,C,L), where B is the batch size,
        C is the number of channels, and L is the length of the sequence.

        Returns:
        - x (torch.Tensor): Output tensor from the convolutional operation of size (B, out_channels, L + padding),
        where out_channels is the number of output channels and L is the length of the sequence.
        """
        x = torch.nn.functional.pad(x, [self.padding, 0])
        x = self.conv(x)
        return x


class ResBlockOneMasked(nn.Module):

    def __init__(self, channels: 'int', kernel_size: 'int', dilation: 'int'):
        """
        Initialize a ResBlockOneMasked object.

        Args:
        - channels (int): the number of input channels.
        - kernel_size (int): the size of the convolutional kernel.
        - dilation (int): the dilation factor of the convolutional kernel.
        """
        super().__init__()
        mid_channels = channels // 2
        self.layer_norm1 = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, mid_channels, kernel_size=1)
        self.layer_norm2 = LayerNorm(mid_channels)
        self.conv2 = MaskedConv1d(mid_channels, mid_channels, kernel_size=kernel_size, dilation=dilation)
        self.layer_norm3 = LayerNorm(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, channels, kernel_size=1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Forward pass of the ResBlockOneMasked object.

        Args:
        - x (torch.Tensor): input tensor with size (B, C, L).

        Returns:
        - y (torch.Tensor): output tensor with size (B, C, L).
        """
        y = x
        y = torch.relu(self.layer_norm1(y))
        y = self.conv1(y)
        y = torch.relu(self.layer_norm2(y))
        y = self.conv2(y)
        y = torch.relu(self.layer_norm3(y))
        y = self.conv3(y)
        return y + x


class ResBlockTwoMasked(nn.Module):

    def __init__(self, channels: 'int', kernel_size: 'int', dilation: 'int'):
        """
        A residual block with two masked convolutions and layer normalization.

        Args:
            channels (int): Number of channels
            kernel_size (int): Size of the convolving kernel
            dilation (int): Spacing between kernel elements
        """
        super().__init__()
        self.conv1 = MaskedConv1d(channels, channels, kernel_size, dilation)
        self.layer_norm1 = LayerNorm(channels)
        self.conv2 = MaskedConv1d(channels, channels, kernel_size, 2 * dilation)
        self.layer_norm2 = LayerNorm(channels)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            y (torch.Tensor): Output tensor
        """
        y = x
        y = self.conv1(y)
        y = torch.relu(self.layer_norm1(y))
        y = self.conv2(y)
        y = torch.relu(self.layer_norm2(y))
        return y + x


class NextItNetLayer(nn.Module):

    def __init__(self, channels: 'int', dilations: 'List[int]', one_masked: 'bool', kernel_size: 'int', feat_drop: 'float'=0.0):
        """
        Args:
            channels: Number of input channels
            dilations: List of dilation sizes for each residual block
            one_masked: Whether to use one-mask convolutions
            kernel_size: Size of convolutional kernel
            feat_drop: Dropout probability for input features, default 0.0
        """
        super().__init__()
        if one_masked:
            ResBlock = ResBlockOneMasked
            if dilations is None:
                dilations = [1, 2, 4]
        else:
            ResBlock = ResBlockTwoMasked
            if dilations is None:
                dilations = [1, 4]
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.res_blocks = nn.ModuleList([ResBlock(channels, kernel_size, dilation) for dilation in dilations])

    def forward(self, emb_seqs, lens):
        """
        Args:
            emb_seqs: Input sequence of embeddings, shape (batch_size, max_len, channels)
            lens: Length of sequences in batch, shape (batch_size,)

        Returns:
            The final state tensor, shape (batch_size, channels)
        """
        batch_size, max_len, _ = emb_seqs.size()
        mask = torch.arange(max_len, device=lens.device).unsqueeze(0).expand(batch_size, max_len) >= lens.unsqueeze(-1)
        emb_seqs = torch.masked_fill(emb_seqs, mask.unsqueeze(-1), 0)
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)
        x = torch.transpose(emb_seqs, 1, 2)
        for res_block in self.res_blocks:
            x = res_block(x)
        batch_idx = torch.arange(len(lens))
        last_idx = lens - 1
        sr = x[batch_idx, :, last_idx]
        return sr


def get_activation(activation: 'str or nn.Module') ->nn.Module:
    """Returns the PyTorch activation function object corresponding to a string.

    Args:
        activation (str or nn.Module): name of the activation function or PyTorch activation function object

    Returns:
        activation_fn (nn.Module): PyTorch activation function object
    """
    if isinstance(activation, str):
        activation_str = activation.lower()
        if activation_str == 'relu':
            activation_fn = nn.ReLU()
        elif activation_str == 'sigmoid':
            activation_fn = nn.Sigmoid()
        elif activation_str == 'tanh':
            activation_fn = nn.Tanh()
        else:
            activation_fn = getattr(nn, activation)()
    else:
        activation_fn = activation
    return activation_fn


class MLP(nn.Module):
    """Customizable Multi-Layer Perceptron"""

    def __init__(self, input_dim: 'int', output_dim: 'Union[int, None]'=None, hidden_units: 'List[int]'=[], hidden_activations: 'Union[str, List[str]]'='ReLU', output_activation: 'Union[str, None]'=None, dropout_rates: 'Union[float, List[float]]'=0.1, batch_norm: 'bool'=False, use_bias: 'bool'=True):
        """Initialize the MLP layer.
        Args:
            input_dim: Size of the input layer.
            output_dim: Size of the output layer (optional).
            hidden_units: List of hidden layer sizes.
            hidden_activations: Activation function for each hidden layer.
            output_activation: Activation function for the output layer (optional).
            dropout_rates: Dropout rate for the hidden layers (same for all if float,
                            otherwise list for individual layers).
            batch_norm: If True, apply batch normalization to each hidden layer.
            use_bias: If True, use bias in each dense layer.
        """
        super(MLP, self).__init__()
        if output_dim is not None:
            assert isinstance(output_dim, int) and output_dim > 0, 'output_dim must be an integer'
        assert isinstance(input_dim, int) and input_dim > 0, 'input_dim must be an integer'
        assert isinstance(hidden_units, list) and all(isinstance(i, int) for i in hidden_units) and len(hidden_units) >= 1, 'hidden_units must be a list of integers and with at least one element'
        if isinstance(hidden_activations, str):
            hidden_activations = [hidden_activations] * len(hidden_units)
        elif isinstance(hidden_activations, list):
            assert len(hidden_activations) == len(hidden_units), 'hidden_activations must have one element per hidden unit'
        else:
            raise TypeError('hidden_activations must be a string or a list of strings')
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        elif isinstance(dropout_rates, list):
            assert len(dropout_rates) == len(hidden_units), 'dropout_rates must have one element per hidden unit'
        else:
            raise TypeError('dropout_rates must be a float or a list of floats')
        hidden_units = [input_dim] + hidden_units
        dense_layers = []
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(get_activation(hidden_activations[idx]))
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.net = nn.Sequential(*dense_layers)

    def forward(self, x):
        """Forward propagate through the neural network.
        Args:
            x: Input tensor with shape (batch_size, input_dim).
        Returns:
            Output tensor with shape (batch_size, output_dim) if output_dim is not None,
            otherwise tensor with shape (batch_size, hidden_units[-1]).
        """
        return self.net(x)


class GraphLayer(nn.Module):

    def __init__(self, num_fields, embedding_dim):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class FiGNN_Layer(nn.Module):

    def __init__(self, num_fields, embedding_dim, gnn_layers=3, reuse_graph_layer=False, use_gru=True, use_residual=True, device=None):
        super(FiGNN_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        self.device = device
        if reuse_graph_layer:
            self.gnn = GraphLayer(num_fields, embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(num_fields, embedding_dim) for _ in range(gnn_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embedding_dim * 2, 1, bias=False)

    def build_graph_with_attention(self, feature_emb):
        src_emb = feature_emb[:, self.src_nodes, :]
        dst_emb = feature_emb[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        mask = torch.eye(self.num_fields)
        alpha = alpha.masked_fill(mask.byte(), float('-inf'))
        graph = F.softmax(alpha, dim=-1)
        return graph

    def forward(self, feature_emb):
        g = self.build_graph_with_attention(feature_emb)
        h = feature_emb
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feature_emb
        return h


class SRGNNConv(nn.Module):
    """
    只是一个图，实现公式(1)
    """

    def __init__(self, dim):
        super(SRGNNConv, self).__init__()
        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, g, ego_embedding):
        hidden = self.lin(ego_embedding)
        g.ndata['h'] = hidden
        g.update_all(message_func=fn.u_mul_e('h', 'edge_weight', 'm'), reduce_func=fn.sum(msg='m', out='h'))
        return g.ndata['h']


class SRGNNCell(nn.Module):
    """
    实现公式(2)(3)(4)(5)
    """

    def __init__(self, dim):
        super(SRGNNCell, self).__init__()
        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)
        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

    def forward(self, in_graph, out_graph, hidden):
        input_in = self.incomming_conv(in_graph, hidden)
        input_out = self.outcomming_conv(out_graph, hidden)
        inputs = torch.cat([input_in, input_out], dim=-1)
        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy


class InnerProductLayer(nn.Module):
    """ output: product_sum_pooling (bs x 1),
                Bi_interaction_pooling (bs * dim),
                inner_product (bs x f2/2),
                elementwise_product (bs x f2/2 x emb_dim)
    """

    def __init__(self, num_fields=None, output='product_sum_pooling'):
        super(InnerProductLayer, self).__init__()
        self._output_type = output
        if output not in ['product_sum_pooling', 'Bi_interaction_pooling', 'inner_product', 'elementwise_product']:
            raise ValueError('InnerProductLayer output={} is not supported.'.format(output))
        if num_fields is None:
            if output in ['inner_product', 'elementwise_product']:
                raise ValueError('num_fields is required when InnerProductLayer output={}.'.format(output))
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triange_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.ByteTensor), requires_grad=False)

    def forward(self, feature_emb):
        if self._output_type in ['product_sum_pooling', 'Bi_interaction_pooling']:
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2
            square_of_sum = torch.sum(feature_emb ** 2, dim=1)
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == 'Bi_interaction_pooling':
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == 'elementwise_product':
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == 'inner_product':
            inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
            flat_upper_triange = torch.masked_select(inner_product_matrix, self.upper_triange_mask)
            return flat_upper_triange.view(-1, self.interaction_units)


class BilinearInteractionLayer(nn.Module):

    def __init__(self, num_fields, embedding_dim, bilinear_type='field_interaction'):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == 'field_all':
            self.bilinear_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif self.bilinear_type == 'field_each':
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False) for i in range(num_fields)])
        elif self.bilinear_type == 'field_interaction':
            self.bilinear_layer = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False) for i, j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == 'field_all':
            bilinear_list = [(self.bilinear_layer(v_i) * v_j) for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == 'field_each':
            bilinear_list = [(self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]) for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == 'field_interaction':
            bilinear_list = [(self.bilinear_layer[i](v[0]) * v[1]) for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class HolographicInteractionLayer(nn.Module):

    def __init__(self, num_fields, interaction_type='circular_convolution'):
        super(HolographicInteractionLayer, self).__init__()
        self.interaction_type = interaction_type
        if self.interaction_type == 'circular_correlation':
            self.conj_sign = nn.Parameter(torch.tensor([1.0, -1.0]), requires_grad=False)
        p, q = zip(*list(combinations(range(num_fields), 2)))
        self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
        self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)

    def forward(self, feature_emb):
        emb1 = torch.index_select(feature_emb, 1, self.field_p)
        emb2 = torch.index_select(feature_emb, 1, self.field_q)
        if self.interaction_type == 'hadamard_product':
            interact_tensor = emb1 * emb2
        elif self.interaction_type == 'circular_convolution':
            fft1 = torch.rfft(emb1, 1, onesided=False)
            fft2 = torch.rfft(emb2, 1, onesided=False)
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim=-1)
            interact_tensor = torch.irfft(fft_product, 1, onesided=False)
        elif self.interaction_type == 'circular_correlation':
            fft1_emb = torch.rfft(emb1, 1, onesided=False)
            fft1 = fft1_emb * self.conj_sign.expand_as(fft1_emb)
            fft2 = torch.rfft(emb2, 1, onesided=False)
            fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim=-1)
            interact_tensor = torch.irfft(fft_product, 1, onesided=False)
        else:
            raise ValueError('interaction_type={} not supported.'.format(self.interaction_type))
        return interact_tensor


class CrossInteractionLayer(nn.Module):

    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


class CrossNet(nn.Module):

    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteractionLayer(input_dim) for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class CompressedInteractionNet(nn.Module):

    def __init__(self, num_fields, cin_layer_units, output_dim=1):
        super(CompressedInteractionNet, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.fc = nn.Linear(sum(cin_layer_units), output_dim)
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer['layer_' + str(i + 1)] = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature_emb):
        pooling_outputs = []
        X_0 = feature_emb
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum('bhd,bmd->bhmd', X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer['layer_' + str(i + 1)](hadamard_tensor).view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)
        output = self.fc(concate_vec)
        return output


class InteractionMachine(nn.Module):

    def __init__(self, embedding_dim, order=2, batch_norm=False):
        super(InteractionMachine, self).__init__()
        assert order < 6, 'order={} is not supported.'.format(order)
        self.order = order
        self.bn = nn.BatchNorm1d(embedding_dim * order) if batch_norm else None
        self.fc = nn.Linear(order * embedding_dim, 1)

    def second_order(self, p1, p2):
        return (p1.pow(2) - p2) / 2

    def third_order(self, p1, p2, p3):
        return (p1.pow(3) - 3 * p1 * p2 + 2 * p3) / 6

    def fourth_order(self, p1, p2, p3, p4):
        return (p1.pow(4) - 6 * p1.pow(2) * p2 + 3 * p2.pow(2) + 8 * p1 * p3 - 6 * p4) / 24

    def fifth_order(self, p1, p2, p3, p4, p5):
        return (p1.pow(5) - 10 * p1.pow(3) * p2 + 20 * p1.pow(2) * p3 - 30 * p1 * p4 - 20 * p2 * p3 + 15 * p1 * p2.pow(2) + 24 * p5) / 120

    def forward(self, X):
        out = []
        Q = X
        if self.order >= 1:
            p1 = Q.sum(dim=1)
            out.append(p1)
            if self.order >= 2:
                Q = Q * X
                p2 = Q.sum(dim=1)
                out.append(self.second_order(p1, p2))
                if self.order >= 3:
                    Q = Q * X
                    p3 = Q.sum(dim=1)
                    out.append(self.third_order(p1, p2, p3))
                    if self.order >= 4:
                        Q = Q * X
                        p4 = Q.sum(dim=1)
                        out.append(self.fourth_order(p1, p2, p3, p4))
                        if self.order == 5:
                            Q = Q * X
                            p5 = Q.sum(dim=1)
                            out.append(self.fifth_order(p1, p2, p3, p4, p5))
        out = torch.cat(out, dim=-1)
        if self.bn is not None:
            out = self.bn(out)
        y = self.fc(out)
        return y


class FM_Layer(nn.Module):

    def __init__(self, final_activation=None, use_bias=True):
        super(FM_Layer, self).__init__()
        self.inner_product_layer = InnerProductLayer(output='product_sum_pooling')
        self.final_activation = final_activation

    def forward(self, feature_emb_list):
        output = self.inner_product_layer(feature_emb_list)
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output


class SENET_Layer(nn.Module):

    def __init__(self, num_fields, reduction_ratio=3):
        super(SENET_Layer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False), nn.ReLU(), nn.Linear(reduced_size, num_fields, bias=False), nn.ReLU())

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class MaskBlock(torch.nn.Module):

    def __init__(self, input_dim: 'int', mask_input_dim: 'int', output_size: 'int', reduction_factor: 'float') ->None:
        """
       Initializes the MaskBlock module.

       Args:
           input_dim (int): The size of the input tensor.
           mask_input_dim (int): The size of the mask tensor.
           output_size (int): The size of the output tensor.
           reduction_factor (float): The factor by which to reduce the size of the mask tensor.
       """
        super(MaskBlock, self).__init__()
        self._input_layer_norm = torch.nn.LayerNorm(input_dim)
        aggregation_size = int(mask_input_dim * reduction_factor)
        self._mask_layer = torch.nn.Sequential(torch.nn.Linear(mask_input_dim, aggregation_size), torch.nn.ReLU(), torch.nn.Linear(aggregation_size, input_dim))
        self._hidden_layer = torch.nn.Linear(input_dim, output_size)
        self._layer_norm = torch.nn.LayerNorm(output_size)

    def forward(self, net: 'torch.Tensor', mask_input: 'torch.Tensor'):
        if self._input_layer_norm:
            net = self._input_layer_norm(net)
        hidden_layer_output = self._hidden_layer(net * self._mask_layer(mask_input))
        return self._layer_norm(hidden_layer_output)


class MultiInterestSelfAttention(nn.Module):

    def __init__(self, embedding_dim: 'int', num_attention_heads: 'int', d: 'int'=None) ->None:
        super(MultiInterestSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        if d is None:
            self.d = self.embedding_dim * 4
        else:
            self.d = d
        self.W1 = nn.Parameter(torch.rand(self.embedding_dim, self.d), requires_grad=True)
        self.W2 = nn.Parameter(torch.rand(self.d, self.num_attention_heads), requires_grad=True)
        self.W3 = nn.Parameter(torch.rand(self.embedding_dim, self.embedding_dim), requires_grad=True)

    def forward(self, sequence_embeddings: 'torch.Tensor', mask: 'torch.Tensor'=None) ->torch.Tensor:
        """
        Args:
            * sequence_embeddings (torch.Tensor): batch_size x sequence_length x embedding_dimension
            * mask (torch.Tensor): binary mask for sequence; batch_size x sequence_length x 1

        Returns:
            * Multi-interest embeddings (torch.Tensor): batch_size x num_attention_heads x embedding_dimension
        """
        H = torch.einsum('bse, ed -> bsd', sequence_embeddings, self.W1).tanh()
        if mask is not None:
            A = torch.einsum('bsd, dk -> bsk', H, self.W2) + -1000000000.0 * (1 - mask.float())
            A = F.softmax(A, dim=1)
        else:
            A = F.softmax(torch.einsum('bsd, dk -> bsk', H, self.W2), dim=1)
        A = A.permute(0, 2, 1)
        multi_interest_emb = torch.matmul(A, sequence_embeddings)
        return multi_interest_emb


class CapsuleNetwork(nn.Module):

    def __init__(self, hidden_size: 'int', seq_len: 'int', bilinear_type: 'int'=2, interest_num: 'int'=4, routing_times: 'int'=3, hard_readout: 'bool'=True, relu_layer: 'bool'=False) ->None:
        """
        Implements a Capsule Network that is capable of handling various types of bilinear
        interactions between items in a sequence.

        Args:
        hidden_size: An integer representing the size of the hidden layer of the model.
        seq_len: An integer representing the length of the input sequence.
        bilinear_type: An integer representing the type of bilinear interaction between items.
        interest_num: An integer representing the number of interest capsules in the model.
        routing_times: An integer representing the number of dynamic routing iterations.
        hard_readout: A Boolean indicating whether to use hard readout or not
        relu_layer: A Boolean indicating whether to use a ReLU layer

        Returns:
        interest_capsule: The output interest capsule from the model
        """
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = False
        self.relu = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.ReLU())
        if self.bilinear_type == 0:
            self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.hidden_size, self.hidden_size * self.interest_num, bias=False)
        else:
            self.w = nn.Parameter(torch.Tensor(1, self.seq_len, self.interest_num * self.hidden_size, self.hidden_size))

    def forward(self, item_eb, mask, device):
        if self.bilinear_type == 0:
            item_eb_hat = self.linear(item_eb)
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:
            u = torch.unsqueeze(item_eb, dim=2)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u, dim=3)
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.hidden_size))
        if self.stop_grad:
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat
        if self.bilinear_type > 0:
            capsule_weight = torch.zeros(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=device, requires_grad=False)
        else:
            capsule_weight = torch.randn(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=device, requires_grad=False)
        for i in range(self.routing_times):
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)
            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)
            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-09)
                interest_capsule = scalar_factor * interest_capsule
                delta_weight = torch.matmul(item_eb_hat_iter, torch.transpose(interest_capsule, 2, 3).contiguous())
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-09)
                interest_capsule = scalar_factor * interest_capsule
        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.hidden_size))
        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)
        return interest_capsule


class MaskedAveragePooling(nn.Module):
    """
    This module takes as input an embedding matrix,
    applies masked pooling, i.e. ignores zero-padding,
    and computes the average embedding vector for each input.
    """

    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix: 'torch.Tensor') ->torch.Tensor:
        """
        Computes the average embedding vector.

        Args:
            embedding_matrix (torch.Tensor): Input embedding of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size) representing the averaged embedding vector.
        """
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix != 0).sum(dim=1)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return embedding_vec


class MaskedSumPooling(nn.Module):
    """
    This module takes as input an embedding matrix,
    applies masked pooling, i.e. ignores zero-padding,
    and computes the sum embedding vector for each input.
    """

    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix: 'torch.Tensor') ->torch.Tensor:
        """
        Computes the sum embedding vector.

        Args:
            embedding_matrix (torch.Tensor): Input embedding of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size) representing the summed embedding vector.
        """
        return torch.sum(embedding_matrix, dim=1)


class KMaxPooling(nn.Module):
    """
    This module takes as input an embedding matrix,
    and returns the k-max pooling along the specified axis.
    """

    def __init__(self, k: 'int', dim: 'int'):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, X: 'torch.Tensor') ->torch.Tensor:
        """
        Computes the k-max pooling.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, k, hidden_size) representing the k-max pooled embedding vector.
        """
        index = X.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
        output = X.gather(self.dim, index)
        return output


class STAMPLayer(nn.Module):

    def __init__(self, embedding_dim: 'int', feat_drop: 'float'=0.0):
        """
        Args:
        embedding_dim(int): the input/output dimensions of the STAMPLayer
        feat_drop(float): Dropout rate to be applied to the input features
        """
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_a = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.fc_t = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.attn_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_t = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.attn_s = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_e = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, emb_seqs: 'torch.Tensor', lens: 'torch.Tensor') ->torch.Tensor:
        """
        Applies the STAMP mechanism to a batch of input sequences

        Args:
        emb_seqs(torch.Tensor): Batch of input sequences [batch_size, max_len, embedding_dim]
        lens(torch.Tensor): A tensor of actual sequence lengths for each sequence in the batch [batch_size]

        Returns:
        sr(torch.Tensor): Output scores of the STAMP mechanism [batch_size, embedding_dim]
        """
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)
        batch_size, max_len, _ = emb_seqs.size()
        mask = torch.arange(max_len, device=lens.device).unsqueeze(0).expand(batch_size, max_len) >= lens.unsqueeze(-1)
        emb_seqs = torch.masked_fill(emb_seqs, mask.unsqueeze(-1), 0)
        ms = emb_seqs.sum(dim=1) / lens.unsqueeze(-1)
        xt = emb_seqs[torch.arange(batch_size), lens - 1]
        ei = self.attn_i(emb_seqs)
        et = self.attn_t(xt).unsqueeze(1)
        es = self.attn_s(ms).unsqueeze(1)
        e = self.attn_e(torch.sigmoid(ei + et + es)).squeeze(-1)
        alpha = torch.masked_fill(e, mask, 0)
        alpha = alpha.unsqueeze(-1)
        ma = torch.sum(alpha * emb_seqs, dim=1)
        ha = self.fc_a(ma)
        ht = self.fc_t(xt)
        sr = ha * ht
        return sr


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


class GRU4RecEncoder(nn.Module):

    def __init__(self, emb_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, emb_size, bias=False)

    def forward(self, seq, lengths):
        sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_seq = seq.index_select(dim=0, index=sort_idx)
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths.cpu(), batch_first=True)
        output, hidden = self.rnn(seq_packed, None)
        sort_rnn_vector = self.out(hidden[-1])
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)
        return rnn_vector


class CaserEncoder(nn.Module):

    def __init__(self, emb_size, max_his, num_horizon=16, num_vertical=8, l=5):
        super().__init__()
        self.max_his = max_his
        lengths = [(i + 1) for i in range(l)]
        self.conv_h = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_horizon, kernel_size=(i, emb_size)) for i in lengths])
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=num_vertical, kernel_size=(max_his, 1))
        self.fc_dim_h = num_horizon * len(lengths)
        self.fc_dim_v = num_vertical * emb_size
        fc_dim_in = self.fc_dim_v + self.fc_dim_h
        self.fc = nn.Linear(fc_dim_in, emb_size)

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        pad_len = self.max_his - seq_len
        seq = F.pad(seq, [0, 0, 0, pad_len]).unsqueeze(1)
        out_v = self.conv_v(seq).view(-1, self.fc_dim_v)
        out_hs = list()
        for conv in self.conv_h:
            conv_out = conv(seq).squeeze(3).relu()
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)
        his_vector = self.fc(torch.cat([out_v, out_h], 1))
        return his_vector


class BERT4RecEncoder(nn.Module):

    def __init__(self, emb_size, max_his, num_layers=2, num_heads=2):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads) for _ in range(num_layers)])

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len))
        valid_mask = len_range[None, :] < lengths[:, None]
        position = len_range[None, :] * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors
        attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()
        his_vector = seq[torch.arange(batch_size), lengths - 1]
        return his_vector


def get_feature_num(enc_dict: 'Dict') ->Tuple[int, int]:
    """Get the number of sparse and dense features.

    Args:
        enc_dict (Dict): Encoding dictionary.

    Returns:
        Tuple[int, int]: The number of sparse and dense features.
    """
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense += 1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse += 1
    return num_sparse, num_dense


def get_dnn_input_dim(enc_dict: 'Dict', embedding_dim: 'int') ->int:
    """Get the input dimension for DNN layers.

    Args:
        enc_dict (Dict): Encoding dictionary.
        embedding_dim (int): Embedding dimension.

    Returns:
        int: The input dimension for DNN layers.
    """
    num_sparse, num_dense = get_feature_num(enc_dict)
    return num_sparse * embedding_dim + num_dense


def get_linear_input(enc_dict: 'Dict', data: 'Dict') ->torch.Tensor:
    """Get the input tensor for linear layers.

    Args:
        enc_dict (Dict): Encoding dictionary.
        data (Dict): Data dictionary.

    Returns:
        torch.Tensor: The input tensor for linear layers.
    """
    res_data = []
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            res_data.append(data[col])
    res_data = torch.stack(res_data, axis=1)
    return res_data


class LR_Layer(nn.Module):

    def __init__(self, enc_dict):
        super(LR_Layer, self).__init__()
        self.enc_dict = enc_dict
        self.emb_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=1)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, 1)
        self.fc = nn.Linear(self.dnn_input_dim, 1)

    def forward(self, data):
        sparse_emb = self.emb_layer(data).squeeze(-1)
        dense_input = get_linear_input(self.enc_dict, data)
        dnn_input = torch.cat((sparse_emb, dense_input), dim=1)
        out = self.fc(dnn_input)
        return out


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


class AITM(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, tower_dims: 'List[int]'=[400, 400, 400], drop_prob: 'List[float]'=[0.1, 0.1, 0.1], enc_dict: 'Dict[str, dict]'=None):
        super(AITM, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.tower_dims = tower_dims
        self.drop_prob = drop_prob
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        self.tower_input_size = self.num_sparse_fea * self.embedding_dim
        self.click_tower = MLP(input_dim=self.tower_input_size, hidden_units=self.tower_dims, hidden_activations='relu', dropout_rates=self.drop_prob)
        self.conversion_tower = MLP(input_dim=self.tower_input_size, hidden_units=self.tower_dims, hidden_activations='relu', dropout_rates=self.drop_prob)
        self.attention_layer = MultiHeadSelfAttention(self.tower_dims[-1])
        self.info_layer = nn.Sequential(nn.Linear(tower_dims[-1], tower_dims[-1]), nn.ReLU(), nn.Dropout(drop_prob[-1]))
        self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1), nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1), nn.Sigmoid())
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the AITM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_embedding = self.embedding_layer(data)
        feature_embedding = feature_embedding.flatten(start_dim=1)
        tower_click = self.click_tower(feature_embedding)
        tower_conversion = torch.unsqueeze(self.conversion_tower(feature_embedding), 1)
        info = torch.unsqueeze(self.info_layer(tower_click), 1)
        ait = self.attention_layer(torch.cat([tower_conversion, info], 1))
        ait = torch.sum(ait, dim=1)
        click = torch.squeeze(self.click_layer(tower_click), dim=1)
        conversion = torch.squeeze(self.conversion_layer(ait), dim=1)
        if is_training:
            loss = self.loss(data['task1_label'], click, data['task2_label'], conversion)
            output_dict = {'task1_pred': click, 'task2_pred': conversion, 'loss': loss}
        else:
            output_dict = {'task1_pred': click, 'task2_pred': conversion}
        return output_dict

    def loss(self, click_label, click_pred, conversion_label, conversion_pred, constraint_weight=0.6):
        click_loss = nn.functional.binary_cross_entropy(click_pred, click_label)
        conversion_loss = nn.functional.binary_cross_entropy(conversion_pred, conversion_label)
        label_constraint = torch.maximum(conversion_pred - click_pred, torch.zeros_like(click_label))
        constraint_loss = torch.sum(label_constraint)
        loss = click_loss + conversion_loss + constraint_weight * constraint_loss
        return loss


class ESSM(BaseModel):

    def __init__(self, embedding_dim=40, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(ESSM, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim
        self.ctr_layer = MLP(input_dim=hidden_size, output_dim=1, hidden_units=self.hidden_dim, hidden_activations='relu', dropout_rates=self.dropouts)
        self.cvr_layer = MLP(input_dim=hidden_size, output_dim=1, hidden_units=self.hidden_dim, hidden_activations='relu', dropout_rates=self.dropouts)
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the ESSM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        click = self.sigmoid(self.ctr_layer(hidden))
        conversion = self.sigmoid(self.cvr_layer(hidden))
        pctrcvr = click * conversion
        if is_training:
            loss = self.loss(click, pctrcvr, data)
            output_dict = {'task1_pred': click, 'task2_pred': conversion, 'loss': loss}
        else:
            output_dict = {'task1_pred': click, 'task2_pred': conversion}
        return output_dict

    def loss(self, click, conversion, data, weight=0.5):
        ctr_loss = nn.functional.binary_cross_entropy(click.squeeze(-1), data['task1_label'])
        cvr_loss = nn.functional.binary_cross_entropy(conversion.squeeze(-1), data['task2_label'])
        loss = cvr_loss + weight * ctr_loss
        return loss


class MLMMOE(BaseModel):

    def __init__(self, num_task=2, n_expert=3, embedding_dim=40, mmoe_hidden_dim=128, expert_activation=None, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(MLMMOE, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.mmoe_hidden_dim = mmoe_hidden_dim
        self.expert_activation = expert_activation
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        self.level_gates = [torch.nn.Parameter(torch.rand(n_expert, 1), requires_grad=True) for _ in range(n_expert)]
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())
        self.set_device(device)
        self.apply(self._init_weights)

    def set_device(self, device):
        for i in range(self.num_task):
            self.gates[i] = self.gates[i]
            self.gates_bias[i] = self.gates_bias[i]
        for i in range(self.n_expert):
            self.level_gates[i] = self.level_gates[i]
        None

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the MLMMOE model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        hidden = torch.cat([hidden, dense_fea], axis=-1)
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)
        level_out = []
        for idx, gate in enumerate(self.level_gates):
            gate = nn.Softmax(dim=0)(gate)
            temp_out = torch.einsum('abc, cd -> abd', experts_out, gate)
            level_out.append(temp_out)
        level_out = torch.cat(level_out, axis=-1)
        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum('ab, bc -> ac', hidden, gate)
            if self.gates_bias:
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)
        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)
            weighted_expert_output = level_out * expanded_gate_output.expand_as(level_out)
            outs.append(torch.sum(weighted_expert_output, 2))
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss
        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight is None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i, _ in enumerate(task_outputs):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1), data[f'task{i + 1}_label'])
        return loss


class MMOE(BaseModel):

    def __init__(self, num_task=2, n_expert=3, embedding_dim=40, mmoe_hidden_dim=128, expert_activation=None, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(MMOE, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.mmoe_hidden_dim = mmoe_hidden_dim
        self.expert_activation = expert_activation
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())
        self.set_device(device)
        self.apply(self._init_weights)

    def set_device(self, device):
        for i in range(self.num_task):
            self.gates[i] = self.gates[i]
            self.gates_bias[i] = self.gates_bias[i]
        None

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the MMOE model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        hidden = torch.cat([hidden, dense_fea], axis=-1)
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)
        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum('ab, bc -> ac', hidden, gate)
            if self.gates_bias:
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)
        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(experts_out)
            outs.append(torch.sum(weighted_expert_output, 2))
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss
        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight is None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i, _ in enumerate(task_outputs):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1) + 1e-06, data[f'task{i + 1}_label'])
        return loss


class OMOE(BaseModel):

    def __init__(self, num_task=2, n_expert=3, embedding_dim=40, omoe_hidden_dim=128, expert_activation=None, hidden_dim=[128, 64], dropouts=[0.2, 0.2], enc_dict=None, device=None):
        super(OMOE, self).__init__(enc_dict, embedding_dim)
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.omoe_hidden_dim = omoe_hidden_dim
        self.expert_activation = expert_activation
        self.hidden_dim = hidden_dim
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, omoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(omoe_hidden_dim, n_expert), requires_grad=True)
        self.gate = torch.nn.Parameter(torch.rand(n_expert, 1), requires_grad=True)
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [omoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())
        self.apply(self._init_weights)

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the OMOE model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        hidden = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        hidden = torch.cat([hidden, dense_fea], axis=-1)
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)
        gate = nn.Softmax(dim=0)(self.gate)
        gate_out = torch.einsum('abc, cd -> abd', experts_out, gate).squeeze(-1)
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = gate_out
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss
        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight is None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i, _ in enumerate(task_outputs):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1), data[f'task{i + 1}_label'])
        return loss


class ShareBottom(BaseModel):

    def __init__(self, num_task: 'int'=2, embedding_dim: 'int'=40, hidden_units: 'List[int]'=[128, 64], dropouts: 'List[float]'=[0.2, 0.2], enc_dict: 'Dict[str, dict]'=None):
        super(ShareBottom, self).__init__(enc_dict, embedding_dim)
        """
        ShareBottom model.

        Args:
            num_task (int): The number of tasks to be performed. Default is 2.
            embedding_dim (int): The size of the embedding vector. Default is 40.
            hidden_units (List[int]): The list of hidden units for each layer. Default is [128, 64].
            dropouts (List[float]): The list of dropout rates for each layer. Default is [0.2, 0.2].
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        self.enc_dict = enc_dict
        self.num_task = num_task
        self.hidden_dim = hidden_units
        self.dropouts = dropouts
        self.num_sparse_fea, self.num_dense_fea = get_feature_num(self.enc_dict)
        hidden_size = self.num_sparse_fea * self.embedding_dim + self.num_dense_fea
        self.apply(self._init_weights)
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [hidden_size] + hidden_units
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer', nn.Linear(hid_dim[-1], 1))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_sigmoid', nn.Sigmoid())

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the ShareBottom model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data).flatten(start_dim=1)
        dense_fea = get_linear_input(self.enc_dict, data)
        out = torch.cat([feature_emb, dense_fea], axis=-1)
        output_dict = dict()
        task_outputs = list()
        for i in range(self.num_task):
            x = out
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
            output_dict[f'task{i + 1}_pred'] = x
        if is_training:
            loss = self.loss(task_outputs, data)
            output_dict['loss'] = loss
        return output_dict

    def loss(self, task_outputs, data, weight=None):
        if weight is None:
            weight = np.ones(self.num_task) / self.num_task
        loss = 0
        for i, _ in enumerate(task_outputs):
            loss += weight[i] * nn.functional.binary_cross_entropy(task_outputs[i].squeeze(-1), data[f'task{i + 1}_label'])
        return loss


class AFM(BaseModel):

    def __init__(self, embedding_dim=32, hidden_units=[64, 64, 64], loss_fun='torch.nn.BCELoss()', enc_dict=None):
        super(AFM, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.senet_layer = SENET_Layer(self.num_sparse, 3)
        self.bilinear_interaction = BilinearInteractionLayer(self.num_sparse, embedding_dim, 'field_interaction')
        input_dim = self.num_sparse * (self.num_sparse - 1) * self.embedding_dim + self.num_dense
        self.dnn = MLP(input_dim=input_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.reset_parameters()

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the AFM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        y_pred = self.lr(data)
        feature_emb = self.embedding_layer(data)
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction(feature_emb)
        bilinear_q = self.bilinear_interaction(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)
        comb_out = torch.cat([comb_out, dense_input], dim=1)
        y_pred += self.dnn(comb_out)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class AFN(BaseModel):

    def __init__(self, embedding_dim=32, dnn_hidden_units=[64, 64, 64], afn_hidden_units=[64, 64, 64], ensemble_dnn=True, loss_fun='torch.nn.BCELoss()', logarithmic_neurons=5, enc_dict=None):
        super(AFN, self).__init__(enc_dict, embedding_dim)
        """
        AFN model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            dnn_hidden_units (List[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            afn_hidden_units (List[int]): The list of hidden units for the AFN. Default is [64, 64, 64].
            ensemble_dnn (bool): Whether to use ensemble DNN. Default is True.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            logarithmic_neurons (int): The number of logarithmic neurons. Default is 5.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """
        self.dnn_hidden_units = dnn_hidden_units
        self.afn_hidden_units = afn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.coefficient_W = nn.Linear(self.num_sparse, logarithmic_neurons, bias=False)
        self.dense_layer = MLP(input_dim=embedding_dim * logarithmic_neurons, output_dim=1, hidden_units=afn_hidden_units, use_bias=True)
        self.log_batch_norm = nn.BatchNorm1d(self.num_sparse)
        self.exp_batch_norm = nn.BatchNorm1d(logarithmic_neurons)
        self.ensemble_dnn = ensemble_dnn
        if ensemble_dnn:
            self.embedding_layer2 = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
            self.dnn = MLP(input_dim=embedding_dim * self.num_sparse, output_dim=1, hidden_units=dnn_hidden_units, use_bias=True)
            self.fc = nn.Linear(2, 1)
        self.reset_parameters()

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the AFN model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        dnn_input = self.logarithmic_net(feature_emb)
        afn_out = self.dense_layer(dnn_input)
        if self.ensemble_dnn:
            feature_emb2 = self.embedding_layer2(data)
            dnn_out = self.dnn(feature_emb2.flatten(start_dim=1))
            y_pred = self.fc(torch.cat([afn_out, dnn_out], dim=-1))
        else:
            y_pred = afn_out
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict

    def logarithmic_net(self, feature_emb):
        feature_emb = torch.abs(feature_emb)
        feature_emb = torch.clamp(feature_emb, min=1e-05)
        log_feature_emb = torch.log(feature_emb)
        log_feature_emb = self.log_batch_norm(log_feature_emb)
        logarithmic_out = self.coefficient_W(log_feature_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(logarithmic_out)
        cross_out = self.exp_batch_norm(cross_out)
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out


class GeneralizedInteraction(nn.Module):

    def __init__(self, input_subspaces, output_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteraction, self).__init__()
        self.input_subspaces = input_subspaces
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.eye(embedding_dim, embedding_dim).unsqueeze(0).repeat(output_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(input_subspaces * num_fields, output_subspaces))
        self.h = nn.Parameter(torch.ones(output_subspaces, embedding_dim, 1))

    def forward(self, B_0, B_i):
        outer_product = torch.einsum('bnh,bnd->bnhd', B_0.repeat(1, self.input_subspaces, 1), B_i.repeat(1, 1, self.num_fields).view(B_i.size(0), -1, self.embedding_dim))
        fusion = torch.matmul(outer_product.permute(0, 2, 3, 1), self.alpha)
        fusion = self.W * fusion.permute(0, 3, 1, 2)
        B_i = torch.matmul(fusion, self.h).squeeze(-1)
        return B_i


class GeneralizedInteractionNet(nn.Module):

    def __init__(self, num_layers, num_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteractionNet, self).__init__()
        self.layers = nn.ModuleList([GeneralizedInteraction(num_fields if i == 0 else num_subspaces, num_subspaces, num_fields, embedding_dim) for i in range(num_layers)])

    def forward(self, B_0):
        B_i = B_0
        for layer in self.layers:
            B_i = layer(B_0, B_i)
        return B_i


class AOANet(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, dnn_hidden_units: 'List[int]'=[64, 64, 64], num_interaction_layers: 'int'=3, num_subspaces: 'int'=4, loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        super(AOANet, self).__init__(enc_dict, embedding_dim)
        """
        AOANet model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            dnn_hidden_units (List[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            num_interaction_layers (int): The number of interaction layers in the Generalized Interaction Net. Default is 3.
            num_subspaces (int): The number of subspaces for the interaction layer. Default is 4.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.dnn = MLP(input_dim=self.embedding_dim * self.num_sparse + self.num_dense, output_dim=None, hidden_units=self.dnn_hidden_units)
        self.gin = GeneralizedInteractionNet(num_interaction_layers, num_subspaces, self.num_sparse, self.embedding_dim)
        self.fc = nn.Linear(dnn_hidden_units[-1] + num_subspaces * self.embedding_dim, 1)
        self.reset_parameters()

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the AoaNet model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        emb_flatten = feature_emb.flatten(start_dim=1)
        dnn_out = self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
        interact_out = self.gin(feature_emb).flatten(start_dim=1)
        y_pred = self.fc(torch.cat([dnn_out, interact_out], dim=-1))
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class AutoInt(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, dnn_hidden_units: 'List[int]'=[64, 64, 64], attention_layers: 'int'=1, num_heads: 'int'=1, attention_dim: 'int'=8, loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        super(AutoInt, self).__init__(enc_dict, embedding_dim)
        """
        AutoInt model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            dnn_hidden_units (List[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            attention_layers (int): The number of attention layers. Default is 1.
            num_heads (int): The number of attention heads. Default is 1.
            attention_dim (int): The dimension of the attention layer. Default is 8.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.lr_layer = LR_Layer(enc_dict=enc_dict)
        self.dnn = MLP(input_dim=self.embedding_dim * self.num_sparse + self.num_dense, output_dim=1, hidden_units=self.dnn_hidden_units)
        self.self_attention = nn.Sequential(*[MultiHeadSelfAttention(self.embedding_dim if i == 0 else num_heads * attention_dim, attention_dim=attention_dim, num_heads=num_heads, align_to='output') for i in range(attention_layers)])
        self.fc = nn.Linear(self.num_sparse * attention_dim * num_heads, 1)
        self.reset_parameters()

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the AutoInt model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        attention_out = self.self_attention(feature_emb)
        attention_out = attention_out.flatten(start_dim=1)
        y_pred = self.fc(attention_out)
        if self.dnn is not None:
            dense_input = get_linear_input(self.enc_dict, data)
            emb_flatten = feature_emb.flatten(start_dim=1)
            y_pred += self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
        if self.lr_layer is not None:
            y_pred += self.lr_layer(data)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class CCPM_ConvLayer(nn.Module):
    """
    Input X: tensor of shape (batch_size, 1, num_fields, embedding_dim)
    """

    def __init__(self, num_fields, channels=[3], kernel_heights=[3], activation='Tanh'):
        super(CCPM_ConvLayer, self).__init__()
        if not isinstance(kernel_heights, list):
            kernel_heights = [kernel_heights] * len(channels)
        elif len(kernel_heights) != len(channels):
            raise ValueError('channels={} and kernel_heights={} should have the same length.'.format(channels, kernel_heights))
        module_list = []
        self.channels = [1] + channels
        layers = len(kernel_heights)
        for i in range(1, len(self.channels)):
            in_channels = self.channels[i - 1]
            out_channels = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            module_list.append(nn.ZeroPad2d((0, 0, kernel_height - 1, kernel_height - 1)))
            module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, 1)))
            if i < layers:
                k = max(3, int((1 - pow(float(i) / layers, layers - i)) * num_fields))
            else:
                k = 3
            module_list.append(KMaxPooling(k, dim=2))
            module_list.append(get_activation(activation))
        self.conv_layer = nn.Sequential(*module_list)

    def forward(self, X):
        return self.conv_layer(X)


class CCPM(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, hidden_units: 'List[int]'=[64, 64, 64], channels: 'List[int]'=[4, 4, 2], kernel_heights: 'List[int]'=[6, 5, 3], loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        super(CCPM, self).__init__(enc_dict, embedding_dim)
        """
        Convolutional Click Prediction Model (CCPM) model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            hidden_units (list[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            channels (List[int]): convolution neural network's kernel size.
            kernel_heights (List[int]): convolution neural network's kernel size.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        self.dnn_hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.conv_layer = CCPM_ConvLayer(self.num_sparse, channels=channels, kernel_heights=kernel_heights)
        conv_out_dim = 3 * embedding_dim * channels[-1]
        self.fc = nn.Linear(conv_out_dim, 1)
        self.reset_parameters()

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the CCPM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        conv_in = torch.unsqueeze(feature_emb, 1)
        conv_out = self.conv_layer(conv_in)
        flatten_out = torch.flatten(conv_out, start_dim=1)
        y_pred = self.fc(flatten_out)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class DCN(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, hidden_units: 'List[int]'=[64, 64, 64], crossing_layers: 'int'=3, loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        super(DCN, self).__init__(enc_dict, embedding_dim)
        """
        Deep & Cross Network (DCN) model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            hidden_units (list[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            crossing_layers (int): num of cross layers in DCN Model.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        self.dnn_hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        input_dim = self.num_sparse * self.embedding_dim + self.num_dense
        self.crossnet = CrossNet(input_dim, crossing_layers)
        self.fc = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the DCN model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(torch.cat([flat_feature_emb, dense_input], dim=1))
        y_pred = self.fc(cross_out).sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class DeepFM(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, hidden_units: 'List[int]'=[64, 64, 64], loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        super(DeepFM, self).__init__(enc_dict, embedding_dim)
        """
        DeepFM model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            hidden_units (list[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.fm = FM_Layer()
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.reset_parameters()

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the DeepFM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        sparse_embedding = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        fm_out = self.fm(sparse_embedding)
        emb_flatten = sparse_embedding.flatten(start_dim=1)
        dnn_input = torch.cat((emb_flatten, dense_input), dim=1)
        dnn_output = self.dnn(dnn_input)
        y_pred = torch.sigmoid(fm_out + dnn_output)
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class FiBiNet(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, hidden_units: 'List[int]'=[64, 64, 64], loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        """
        FiBiNet model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            hidden_units (list[int]): The list of hidden units for the DNN. Default is [64, 64, 64].
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        super(FiBiNet, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.senet_layer = SENET_Layer(self.num_sparse, 3)
        self.bilinear_interaction = BilinearInteractionLayer(self.num_sparse, embedding_dim, 'field_interaction')
        input_dim = self.num_sparse * (self.num_sparse - 1) * self.embedding_dim + self.num_dense
        self.dnn = MLP(input_dim=input_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.Tensor]', is_training: 'bool'=True) ->Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the FiBiNet model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        y_pred = self.lr(data)
        feature_emb = self.embedding_layer(data)
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction(feature_emb)
        bilinear_q = self.bilinear_interaction(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)
        comb_out = torch.cat([comb_out, dense_input], dim=1)
        y_pred += self.dnn(comb_out)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class FM(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        """
        Factorization Machine (FM) model.

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """
        super(FM, self).__init__(enc_dict, embedding_dim)
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.fm = FM_Layer()
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.Tensor]', is_training: 'bool'=True) ->Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the FM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        y_pred = self.fm(feature_emb)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class LR(nn.Module):

    def __init__(self, loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        """
        Logistic Regression (LR) model.

        Args:
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, int]): The dictionary containing the encoding information for the features.
        """
        super(LR, self).__init__()
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.lr_layer = LR_Layer(enc_dict=self.enc_dict)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.Tensor]', is_training: 'bool'=True) ->Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the LR model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        y_pred = self.lr_layer(data)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class MaskNet(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, block_num: 'int'=3, use_parallel: 'bool'=True, reduction_factor: 'float'=0.3, hidden_units: 'List[int]'=[64, 64, 64], loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        super(MaskNet, self).__init__(enc_dict, embedding_dim)
        """A class for the MaskNet

        Args:
            embedding_dim (int): The size of the embedding vector. Default is 32.
            block_num (int): The number of MaskBlocks to use. Default is 3.
            use_parallel (bool): If True, use parallel processing for the MaskBlocks. Default is True.
            reduction_factor (float): The reduction factor used to scale the output size of the MaskBlocks. Default is 0.3.
            hidden_units (List[int]): A list of integers representing the number of hidden units in each layer of the MLP. Default is [64, 64, 64].
            loss_fun (str): The loss function used for training. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): The dictionary containing the encoding information for the features.
        """
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.block_num = block_num
        self.hidden_units = hidden_units
        self.reduction_factor = reduction_factor
        self.use_parallel = use_parallel
        self.block_output_dim = self.mask_input_dim = self.input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.mask_block_list = torch.nn.ModuleList()
        for _ in range(self.block_num):
            self.mask_block_list.append(MaskBlock(self.input_dim, self.mask_input_dim, self.block_output_dim, self.reduction_factor))
        self.mlp = MLP(self.block_output_dim, hidden_units=self.hidden_units, output_dim=1)
        self.reset_parameters()

    def forward(self, data, is_training=True):
        """
        Perform forward propagation on the MaskNet model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        sparse_embedding = self.embedding_layer(data)
        dense_input = get_linear_input(self.enc_dict, data)
        emb_flatten = sparse_embedding.flatten(start_dim=1)
        dnn_input = torch.cat((emb_flatten, dense_input), dim=1)
        if self.use_parallel:
            mask_outputs = []
            for layer in self.mask_block_list:
                mask_outputs.append(layer(dnn_input, dnn_input))
            mask_outputs = torch.stack(mask_outputs, dim=1)
            mask_output = torch.mean(mask_outputs, dim=1)
        else:
            mask_output = dnn_input
            for layer in self.mask_block_list:
                mask_output = layer(mask_output, dnn_input)
        y_pred = self.mlp(mask_output).sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class NFM(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, hidden_units: 'List[int]'=[64, 64, 64], loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None):
        """
        Neural Factorization Machine (NFM) model.

        Args:
            embedding_dim (int, optional): The dimension of the embedding layer. Defaults to 32.
            hidden_units (List[int], optional): The number of hidden units in the DNN layers. Defaults to [64, 64, 64].
            loss_fun (str, optional): The loss function. Defaults to 'torch.nn.BCELoss()'.
            enc_dict (Optional[Dict[str, int]], optional): The encoding dictionary. Defaults to None.
        """
        super(NFM, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.inner_product_layer = InnerProductLayer(output='Bi_interaction_pooling')
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP(input_dim=self.embedding_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.Tensor]', is_training: 'bool'=True) ->Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the NFM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        y_pred = self.lr(data)
        batch_size = y_pred.shape[0]
        sparse_embedding = self.embedding_layer(data)
        inner_product_tensor = self.inner_product_layer(sparse_embedding)
        bi_pooling_tensor = inner_product_tensor.view(batch_size, -1)
        y_pred += self.dnn(bi_pooling_tensor)
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class WDL(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, hidden_units: 'List[int]'=[64, 64, 64], loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None) ->None:
        """
        Wide and Deep (WDL) Model

        Args:
            embedding_dim (int): Dimension of the embedding vectors. Defaults to 32.
            hidden_units (list): Number of units in each hidden layer of the MLP. Defaults to [64, 64, 64].
            loss_fun (str): String representation of the loss function. Defaults to 'torch.nn.BCELoss()'.
            enc_dict (dict): Dictionary for encoding input features.
        """
        super(WDL, self).__init__(enc_dict, embedding_dim)
        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.lr = LR_Layer(enc_dict=self.enc_dict)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units, hidden_activations='relu', dropout_rates=0)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.Tensor]', is_training: 'bool'=True) ->Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the WDL model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        wide_logit = self.lr(data)
        sparse_emb = self.embedding_layer(data)
        sparse_emb = sparse_emb.flatten(start_dim=1)
        dense_input = get_linear_input(self.enc_dict, data)
        dnn_input = torch.cat([sparse_emb, dense_input], dim=1)
        deep_logit = self.dnn(dnn_input)
        y_pred = (wide_logit + deep_logit).sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class xDeepFM(BaseModel):

    def __init__(self, embedding_dim: 'int'=32, dnn_hidden_units: 'List[int]'=[64, 64, 64], cin_layer_units: 'List[int]'=[16, 16, 16], loss_fun: 'str'='torch.nn.BCELoss()', enc_dict: 'Dict[str, dict]'=None) ->None:
        """
        xDeepFM model.

        Args:
            embedding_dim (int): The dimension of the embedding vector. Default is 32.
            dnn_hidden_units (List[int]): The number of units in the MLP hidden layers. Default is [64, 64, 64].
            cin_layer_units (List[int]): The number of units in the CIN layers. Default is [16, 16, 16].
            loss_fun (str): String representation of the loss function. Default is 'torch.nn.BCELoss()'.
            enc_dict (Dict[str, dict]): Dictionary for encoding input features.
        """
        super(xDeepFM, self).__init__(enc_dict, embedding_dim)
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)
        self.dnn = MLP(input_dim=self.num_sparse * self.embedding_dim + self.num_dense, output_dim=1, hidden_units=self.dnn_hidden_units)
        self.lr_layer = LR_Layer(enc_dict=self.enc_dict)
        self.cin = CompressedInteractionNet(self.num_sparse, cin_layer_units, output_dim=1)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.Tensor]', is_training: 'bool'=True) ->Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the xDeepFM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        feature_emb = self.embedding_layer(data)
        lr_logit = self.lr_layer(data)
        cin_logit = self.cin(feature_emb)
        if self.dnn is not None:
            dense_input = get_linear_input(self.enc_dict, data)
            emb_flatten = feature_emb.flatten(start_dim=1)
            dnn_logit = self.dnn(torch.cat([emb_flatten, dense_input], dim=1))
            y_pred = lr_logit + cin_logit + dnn_logit
        else:
            y_pred = lr_logit + cin_logit
        y_pred = y_pred.sigmoid()
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}
        return output_dict


class ContraLoss(nn.Module):

    def __init__(self, device, temperature=0.2):
        super(ContraLoss, self).__init__()
        self.device = device
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        If both `labels` and `mask` are None, it degenerates to InfoNCE loss
        Args:
            features: hidden vector of shape [bsz, n_views, dim].
            labels: target item of shape [bsz].
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.transpose(0, 1)).float()
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.transpose(0, 1)) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(mask.shape[0]).view(-1, 1), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        loss = -self.temperature * mean_log_prob_pos
        return loss.mean()


class CLRec(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(CLRec, self).__init__(enc_dict, config)
        self.temp = self.config.get('temp', 0.1)
        self.encoder = BERT4RecEncoder(self.embedding_dim, self.max_length, num_layers=2, num_heads=2)
        self.contra_loss = ContraLoss(temperature=self.temp)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        item_seq_length = torch.sum(mask, dim=1)
        seq_emb = self.item_emb(item_seq)
        user_emb = self.encoder(seq_emb, item_seq_length)
        if is_training:
            item = data['target_item'].squeeze()
            target_item_emb = self.item_emb(item)
            features = torch.stack([user_emb, target_item_emb], dim=1)
            features = F.normalize(features, dim=-1)
            loss = self.calculate_loss(user_emb, item)
            loss += self.contra_loss(features)
            output_dict = {'loss': loss}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


class CMI(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(CMI, self).__init__(enc_dict, config)
        self.hidden_size = self.config.get('hidden_size', 64)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout_prob = self.config.get('dropout_prob', 0)
        self.temp = self.config.get('temp', 0.1)
        self.w_uniform = self.config.get('w_uniform', 1)
        self.w_orth = self.config.get('w_orth', 10)
        self.w_sharp = self.config.get('w_sharp', 1)
        self.w_clloss = self.config.get('w_clloss', 0.05)
        self.n_interest = self.config.get('K', 8)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.W = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.selfatt_W = nn.Linear(self.n_interest, self.n_interest, bias=False)
        self.interest_embedding = nn.Embedding(self.n_interest, self.embedding_dim)
        self.temperature = 0.1
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim, num_layers=self.num_layers, bias=False, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=True), nn.ReLU())
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        with torch.no_grad():
            w = self.item_emb.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.item_emb.weight.copy_(w)
            w = self.interest_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.interest_embedding.weight.copy_(w)
        item_seq = data['hist_item_list']
        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)
        batch_size, n_seq = item_seq.shape
        item_seq_emb = self.item_emb(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)
        psnl_interest = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        interest_cl = self.w_orth * self.get_orth_loss(self.interest_embedding.weight)
        for i in range(1):
            scores = item_seq_emb.matmul(psnl_interest.transpose(1, 2)) / self.temp
            scores = scores.reshape(batch_size * n_seq, -1)
            mask = (item_seq > 0).reshape(-1)
            probs = torch.softmax(scores.reshape(batch_size, n_seq, -1), dim=-1) * (item_seq > 0).float().unsqueeze(-1)
            if self.w_uniform:
                interest_prb_vec = torch.sum(probs.reshape(batch_size * n_seq, -1), dim=0) / torch.sum(mask)
                interest_cl += self.w_uniform * interest_prb_vec.std() / interest_prb_vec.mean()
            psnl_interest = probs.transpose(1, 2).matmul(item_seq_emb)
            psnl_interest = F.normalize(psnl_interest, dim=-1, p=2)
            sys_interest_vec = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            interest_mask = torch.sum(probs, dim=1)
            psnl_interest = torch.where(interest_mask.unsqueeze(-1) > 0, psnl_interest, sys_interest_vec)
            batch_size, seq_len, n_interest = probs.shape
        gru_output, _ = self.gru(item_seq_emb)
        gru_output = self.mlp(gru_output)
        full_psnl_emb = self.gather_indexes(gru_output, item_seq_len - 1)
        full_psnl_emb = F.normalize(full_psnl_emb, p=2, dim=-1)
        psnl_interest = psnl_interest + full_psnl_emb.unsqueeze(1)
        psnl_interest = F.normalize(psnl_interest, p=2, dim=-1)
        if is_training:
            output_dict = {'global_user_emb': full_psnl_emb, 'user_emb': psnl_interest, 'loss': self.calculate_cmi_loss(psnl_interest, data['target_item'].squeeze())}
        else:
            output_dict = {'user_emb': psnl_interest}
        return output_dict

    def get_neg_item(self, batch_size):
        n_item = self.item_emb.weight.shape[0]
        return torch.randint(1, n_item - 1, (batch_size, 1)).squeeze()

    def calculate_cmi_loss(self, psnl_interest, pos_items):
        batch_size, n_interest, embed_size = psnl_interest.shape
        neg_items = self.get_neg_item(batch_size)
        neg_items = neg_items
        pos_items_emb = self.item_emb(pos_items)
        neg_items_emb = self.item_emb(neg_items)
        pos_scores = torch.sum(psnl_interest * pos_items_emb.unsqueeze(1), dim=-1)
        neg_scores = psnl_interest.reshape(-1, embed_size).matmul(neg_items_emb.transpose(0, 1)).reshape(batch_size, -1, batch_size)
        scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        scores = torch.max(scores, dim=1)[0]
        loss = self.loss_fun(scores / self.temp, torch.zeros(batch_size, device=pos_items.device).long())
        multi_clloss = self.multi_inter_clloss(psnl_interest)
        loss += self.w_clloss * multi_clloss
        return loss

    def multi_inter_clloss(self, user_interests):
        """
        下标 0 和 1 是同一个用户数据增强 后所学的不同兴趣，2 和 3 是同一个用户，以此类推，同一个用户同一兴趣之间是正样本，不同用户或不同兴趣之间是负样本
        Args:
            user_interests: batch_size * n_interest * embed_size
        Returns: loss
        """
        device = user_interests.device
        batch_size, n_interest, embed_size = user_interests.shape
        user_interests = user_interests.reshape(batch_size // 2, 2, n_interest, embed_size)
        user_interests_a = user_interests[:, 0].reshape(-1, embed_size)
        user_interests_b = user_interests[:, 1].reshape(-1, embed_size)
        user_interests_a = F.normalize(user_interests_a, p=2, dim=-1)
        user_interests_b = F.normalize(user_interests_b, p=2, dim=-1)
        sim_matrix = user_interests_a.matmul(user_interests_b.transpose(0, 1)) / self.temperature
        loss = F.cross_entropy(sim_matrix, torch.arange(sim_matrix.shape[0], device=device)) + F.cross_entropy(sim_matrix.transpose(0, 1), torch.arange(sim_matrix.shape[0], device=device))
        return loss

    def get_orth_loss(self, x):
        """
        Args:
            x: batch_size * embed_size; Orthogonal embeddings
        Returns:
        """
        num, embed_size = x.shape
        sim = x.reshape(-1, embed_size).matmul(x.reshape(-1, embed_size).transpose(0, 1))
        try:
            diff = sim - torch.eye(sim.shape[1])
        except RuntimeError:
            None
        regloss = diff.pow(2).sum() / (num * num)
        return regloss

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


class ComirecSA(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(ComirecSA, self).__init__(enc_dict, config)
        self.multi_interest_sa = MultiInterestSelfAttention(embedding_dim=self.embedding_dim, num_attention_heads=self.config['K'])
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        if is_training:
            item = data['target_item'].squeeze()
            seq_emb = self.item_emb(item_seq)
            item_e = self.item_emb(item).squeeze(1)
            mask = mask.unsqueeze(-1).float()
            multi_interest_emb = self.multi_interest_sa(seq_emb, mask)
            cos_res = torch.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            k_index = torch.argmax(cos_res, dim=1)
            best_interest_emb = torch.rand(multi_interest_emb.shape[0], multi_interest_emb.shape[2])
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]
            loss = self.calculate_loss(best_interest_emb, item)
            output_dict = {'user_emb': multi_interest_emb, 'loss': loss}
        else:
            seq_emb = self.item_emb(item_seq)
            mask = mask.unsqueeze(-1).float()
            multi_interest_emb = self.multi_interest_sa(seq_emb, mask)
            output_dict = {'user_emb': multi_interest_emb}
        return output_dict


class ComirecDR(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(ComirecDR, self).__init__(enc_dict, config)
        self.capsule = CapsuleNetwork(self.embedding_dim, self.max_length, bilinear_type=2, interest_num=self.config['K'])
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        f"""
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        if is_training:
            item = data['target_item'].squeeze()
            seq_emb = self.item_emb(item_seq)
            item_e = self.item_emb(item).squeeze(1)
            multi_interest_emb = self.capsule(seq_emb, mask, self.device)
            cos_res = torch.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            k_index = torch.argmax(cos_res, dim=1)
            best_interest_emb = torch.rand(multi_interest_emb.shape[0], multi_interest_emb.shape[2])
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]
            loss = self.calculate_loss(best_interest_emb, item)
            output_dict = {'user_emb': multi_interest_emb, 'loss': loss}
        else:
            seq_emb = self.item_emb(item_seq)
            multi_interest_emb = self.capsule(seq_emb, mask, self.device)
            output_dict = {'user_emb': multi_interest_emb}
        return output_dict


class DataAugmenter:

    def __init__(self, num_items, beta_a=3, beta_b=3):
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.num_items = num_items

    def reorder_op(self, seq):
        ratio = torch.distributions.beta.Beta(self.beta_a, self.beta_b).sample().item()
        select_len = int(len(seq) * ratio)
        start = torch.randint(0, len(seq) - select_len + 1, (1,)).item()
        idx_range = torch.arange(len(seq))
        idx_range[start:start + select_len] = idx_range[start:start + select_len][torch.randperm(select_len)]
        return seq[idx_range]

    def mask_op(self, seq):
        ratio = torch.distributions.beta.Beta(self.beta_a, self.beta_b).sample().item()
        selected_len = int(len(seq) * ratio)
        mask = torch.full((len(seq),), False, dtype=torch.bool)
        mask[:selected_len] = True
        mask = mask[torch.randperm(len(mask))]
        seq[mask] = self.num_items
        return seq

    def augment(self, seqs):
        seqs = seqs.clone()
        for i, seq in enumerate(seqs):
            if torch.rand(1) > 0.5:
                seqs[i] = self.mask_op(seq.clone())
            else:
                seqs[i] = self.reorder_op(seq.clone())
        return seqs


class ContraRec(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(ContraRec, self).__init__(enc_dict, config)
        self.gamma = self.config.get('gamma', 1)
        self.beta_a = self.config.get('beta_a', 3)
        self.beta_b = self.config.get('beta_b', 3)
        self.ctc_temp = self.config.get('ctc_temp', 0.2)
        self.ccc_temp = self.config.get('ccc_temp', 0.2)
        self.encoder_name = self.config.get('encoder_name', 'BERT4Rec')
        self.encoder = self.init_encoder(self.encoder_name)
        self.data_augmenter = DataAugmenter(beta_a=self.beta_a, beta_b=self.beta_b, num_items=self.enc_dict[self.config['item_col']]['vocab_size'] - 1)
        self.ccc_loss = ContraLoss(self.device, temperature=self.ccc_temp)
        self.reset_parameters()

    def init_encoder(self, encoder_name):
        if encoder_name == 'GRU4Rec':
            encoder = GRU4RecEncoder(self.embedding_dim, hidden_size=128)
        elif encoder_name == 'Caser':
            encoder = CaserEncoder(self.embedding_dim, self.max_length, num_horizon=16, num_vertical=8, l=5)
        elif encoder_name == 'BERT4Rec':
            encoder = BERT4RecEncoder(self.embedding_dim, self.max_length, num_layers=2, num_heads=2)
        else:
            raise ValueError('Invalid sequence encoder.')
        return encoder

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        item_seq_length = torch.sum(mask, dim=1)
        seq_emb = self.item_emb(item_seq)
        user_emb = self.encoder(seq_emb, item_seq_length)
        if is_training:
            item = data['target_item'].squeeze()
            aug_seq1 = self.data_augmenter.augment(item_seq)
            aug_seq2 = self.data_augmenter.augment(item_seq)
            aug_seq_emb1 = self.item_emb(aug_seq1)
            aug_seq_emb2 = self.item_emb(aug_seq2)
            aug_user_emb1 = self.encoder(aug_seq_emb1, item_seq_length)
            aug_user_emb2 = self.encoder(aug_seq_emb2, item_seq_length)
            features = torch.stack([aug_user_emb1, aug_user_emb2], dim=1)
            features = F.normalize(features, dim=-1)
            loss = self.calculate_loss(user_emb, item) + self.gamma * self.ccc_loss(features, item)
            output_dict = {'loss': loss}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


def pad_sequence(seqs: 'List[torch.Tensor]', max_len: 'int') ->torch.Tensor:
    """Pad sequences to the same length.

    Args:
        seqs (List[torch.Tensor]): List of sequences.
        max_len (int): Maximum length to pad.

    Returns:
        torch.Tensor: Padded sequences tensor.
    """
    padded_seqs = []
    for seq in seqs:
        seq_len = seq.shape[0]
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, dtype=torch.long, device=seq.device)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq[:max_len]
        padded_seqs.append(padded_seq)
    padded_seqs = torch.stack(padded_seqs, dim=0)
    return padded_seqs


def generate_graph(batch_data: 'Dict') ->Dict:
    """Generate session graph.

    Args:
    batch_data (Dict): Batch data dictionary.

    Returns:
        Dict: New batch data dictionary with graph information.
    """
    x = []
    edge_index = []
    alias_inputs = []
    device = batch_data['hist_mask_list'].device
    item_seq_len = torch.sum(batch_data['hist_mask_list'], dim=-1).cpu().numpy()
    """
    # 小例子
    seq = torch.tensor([22,23,21,22,23,23,24,25,21])
    map_index, idx = torch.unique(seq, return_inverse=True)

    print(map_index,idx)
    (tensor([21, 22, 23, 24, 25]), tensor([1, 2, 0, 1, 2, 2, 3, 4, 0]))

    #通过以下方式过去图的emb
    g = item_emb(map_index)

    """
    for i, seq in enumerate(list(torch.chunk(batch_data['hist_item_list'], batch_data['hist_item_list'].shape[0]))):
        seq = seq[seq > 0]
        seq, idx = torch.unique(seq, return_inverse=True)
        x.append(seq)
        alias_seq = idx.squeeze(0)
        alias_inputs.append(alias_seq)
        edge = torch.stack([alias_seq[:-1], alias_seq[1:]])
        edge_index.append(edge)
    """
    对一个batch内的所有session graph进行防冲突处理
    核心逻辑给每个序列的index加上前一个序列的index的最大值，
    保证每个序列在图中对应的节点的index范围互不冲突
    """
    tot_node_num = torch.zeros([1], dtype=torch.long, device=device)
    for i in range(batch_data['hist_item_list'].shape[0]):
        edge_index[i] = edge_index[i] + tot_node_num
        alias_inputs[i] = alias_inputs[i] + tot_node_num
        tot_node_num += x[i].shape[0]
    x = torch.cat(x)
    alias_inputs = pad_sequence(alias_inputs, max_len=batch_data['hist_item_list'].shape[1])
    edge_index = torch.cat(edge_index, dim=1)
    reversed_edge_index = torch.flip(edge_index, dims=[0])
    in_graph = dgl.graph((edge_index[0], edge_index[1]))
    src_degree = in_graph.out_degrees().float()
    norm = torch.pow(src_degree, -1).unsqueeze(1)
    edge_weight = norm[edge_index[0]]
    in_graph.edata['edge_weight'] = edge_weight
    out_graph = dgl.graph((reversed_edge_index[0], reversed_edge_index[1]))
    src_degree = out_graph.out_degrees().float()
    norm = torch.pow(src_degree, -1).unsqueeze(1)
    edge_weight = norm[reversed_edge_index[0]]
    out_graph.edata['edge_weight'] = edge_weight
    new_batch_data = {'x': x, 'alias_inputs': alias_inputs, 'in_graph': in_graph, 'out_graph': out_graph}
    return new_batch_data


class GCSAN(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(GCSAN, self).__init__(enc_dict, config)
        self.n_layers = config.get('n_layers', 2)
        self.n_heads = config.get('n_heads', 4)
        self.hidden_size = config.get('hidden_size', 64)
        self.inner_size = config.get('inner_size', 32)
        self.hidden_dropout_prob = config.get('hidden_dropout_prob', 0.1)
        self.attn_dropout_prob = config.get('attn_dropout_prob', 0.1)
        self.hidden_act = config.get('hidden_act', 'gelu')
        self.layer_norm_eps = config.get('layer_norm_eps', 0.001)
        self.step = config.get('step', 1)
        self.weight = config.get('weight', 0.1)
        self.gnncell = SRGNNCell(self.embedding_dim)
        self.linear_one = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_two = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_three = nn.Linear(self.embedding_dim, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.self_attention = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads, hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob, attn_dropout_prob=self.attn_dropout_prob, hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """
        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)
        batch_data = generate_graph(data)
        hidden = self.item_emb(batch_data['x'])
        for i in range(self.step):
            hidden = self.gnncell(batch_data['in_graph'], batch_data['out_graph'], hidden)
        seq_hidden = hidden[batch_data['alias_inputs']]
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        attention_mask = self.get_attention_mask(data['hist_mask_list'])
        outputs = self.self_attention(seq_hidden, attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        at = self.gather_indexes(output, item_seq_len - 1)
        seq_output = self.weight * at + (1 - self.weight) * ht
        output_dict = dict()
        if is_training:
            loss = self.calculate_loss(seq_output, data['target_item'].squeeze())
            output_dict['loss'] = loss
        else:
            output_dict['user_emb'] = seq_output
        return output_dict


class GRU4Rec(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(GRU4Rec, self).__init__(enc_dict, config)
        self.gru = GRU4RecEncoder(self.embedding_dim, self.embedding_dim)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        item_seq_length = torch.sum(mask, dim=1)
        seq_emb = self.item_emb(item_seq)
        user_emb = self.gru(seq_emb, item_seq_length)
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            output_dict = {'user_emb': user_emb, 'loss': loss}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


class DisentangleEncoder(nn.Module):

    def __init__(self, k_intention, embed_size, max_len):
        super(DisentangleEncoder, self).__init__()
        self.embed_size = embed_size
        self.intentions = nn.Parameter(torch.randn(k_intention, embed_size))
        self.pos_fai = nn.Embedding(max_len, embed_size)
        self.rou = nn.Parameter(torch.randn(embed_size))
        self.W = nn.Linear(embed_size, embed_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.layer_norm_3 = nn.LayerNorm(embed_size)
        self.layer_norm_4 = nn.LayerNorm(embed_size)
        self.layer_norm_5 = nn.LayerNorm(embed_size)

    def forward(self, local_item_emb, global_item_emb, seq_len):
        """
        Args:
            local_item_emb: [B, L, D]
            global_item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            disentangled_intention_emb: [B, K, L, D]
        """
        local_disen_emb = self.intention_disentangling(local_item_emb, seq_len)
        global_siden_emb = self.intention_disentangling(global_item_emb, seq_len)
        disentangled_intention_emb = local_disen_emb + global_siden_emb
        return disentangled_intention_emb

    def item2IntentionScore(self, item_emb):
        """
        Args:
            item_emb: [B, L, D]
        Returns:
            score: [B, L, K]
        """
        item_emb_norm = self.layer_norm_1(item_emb)
        intention_norm = self.layer_norm_2(self.intentions).unsqueeze(0)
        logits = item_emb_norm @ intention_norm.permute(0, 2, 1)
        score = F.softmax(logits / math.sqrt(self.embed_size), -1)
        return score

    def item2AttnWeight(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            score: [B, L]
        """
        B, L = item_emb.size(0), item_emb.size(1)
        dev = item_emb.device
        item_query_row = item_emb[torch.arange(B), seq_len - 1]
        item_query_row += self.pos_fai(seq_len - 1) + self.rou
        item_query = self.layer_norm_3(item_query_row).unsqueeze(1)
        pos_fai_tensor = self.pos_fai(torch.arange(L)).unsqueeze(0)
        item_key_hat = self.layer_norm_4(item_emb + pos_fai_tensor)
        item_key = item_key_hat + torch.relu(self.W(item_key_hat))
        logits = item_query @ item_key.permute(0, 2, 1)
        logits = logits.squeeze() / math.sqrt(self.embed_size)
        score = F.softmax(logits, -1)
        return score

    def intention_disentangling(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B. L, D]
            seq_len: [B]
        Returns:
            item_disentangled_emb: [B, K, L, D]
        """
        item2intention_score = self.item2IntentionScore(item_emb)
        item_attn_weight = self.item2AttnWeight(item_emb, seq_len)
        score_fuse = item2intention_score * item_attn_weight.unsqueeze(-1)
        score_fuse = score_fuse.permute(0, 2, 1).unsqueeze(-1)
        item_emb_k = item_emb.unsqueeze(1)
        disentangled_item_emb = self.layer_norm_5(score_fuse * item_emb_k)
        return disentangled_item_emb


class GlobalSeqEncoder(nn.Module):

    def __init__(self, embed_size, max_len, dropout=0.5):
        super(GlobalSeqEncoder, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.Q_s = nn.Parameter(torch.randn(max_len, embed_size))
        self.K_linear = nn.Linear(embed_size, embed_size)
        self.V_linear = nn.Linear(embed_size, embed_size)

    def forward(self, item_seq, seq_len, item_embeddings):
        """
        Args:
            item_seq (tensor): [B, L]
            seq_len (tensor): [B]
            item_embeddings (tensor): [num_items, D], item embedding table

        Returns:
            global_seq_emb: [B, L, D]
        """
        item_emb = item_embeddings(item_seq)
        item_key = self.K_linear(item_emb)
        item_value = self.V_linear(item_emb)
        attn_logits = self.Q_s @ item_key.permute(0, 2, 1)
        attn_score = F.softmax(attn_logits, -1)
        global_seq_emb = self.dropout(attn_score @ item_value)
        return global_seq_emb


class InfoNCELoss(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss
    """

    def __init__(self, temperature, similarity_type):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.sim_type = similarity_type
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden_view1, aug_hidden_view2, mask=None):
        """
        Args:
            aug_hidden_view1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden_view2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden_view1.ndim > 2:
            aug_hidden_view1 = aug_hidden_view1.view(aug_hidden_view1.size(0), -1)
            aug_hidden_view2 = aug_hidden_view2.view(aug_hidden_view2.size(0), -1)
        if self.sim_type not in ['cos', 'dot']:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. Please choose from ['cos', 'dot']")
        if self.sim_type == 'cos':
            sim11 = self.cosinesim(aug_hidden_view1, aug_hidden_view1)
            sim22 = self.cosinesim(aug_hidden_view2, aug_hidden_view2)
            sim12 = self.cosinesim(aug_hidden_view1, aug_hidden_view2)
        elif self.sim_type == 'dot':
            sim11 = aug_hidden_view1 @ aug_hidden_view1.t()
            sim22 = aug_hidden_view2 @ aug_hidden_view2.t()
            sim12 = aug_hidden_view1 @ aug_hidden_view2.t()
        sim11[..., range(sim11.size(0)), range(sim11.size(0))] = float('-inf')
        sim22[..., range(sim22.size(0)), range(sim22.size(0))] = float('-inf')
        cl_logits1 = torch.cat([sim12, sim11], -1)
        cl_logits2 = torch.cat([sim22, sim12.t()], -1)
        cl_logits = torch.cat([cl_logits1, cl_logits2], 0) / self.temperature
        if mask is not None:
            cl_logits = torch.masked_fill(cl_logits, mask, float('-inf'))
        target = torch.arange(cl_logits.size(0)).long()
        cl_loss = self.criterion(cl_logits, target)
        return cl_loss

    def cosinesim(self, aug_hidden1, aug_hidden2):
        h = torch.matmul(aug_hidden1, aug_hidden2.T)
        h1_norm2 = aug_hidden1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = aug_hidden2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, embed_size, nhead, attn_dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.nhead = nhead
        if self.embed_size % self.nhead != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (self.embed_size, self.nhead))
        self.head_dim = self.embed_size // self.nhead
        self.fc_q = nn.Linear(self.embed_size, self.embed_size)
        self.fc_k = nn.Linear(self.embed_size, self.embed_size)
        self.fc_v = nn.Linear(self.embed_size, self.embed_size)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc_o = nn.Linear(self.embed_size, self.embed_size)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.head_dim).float()))

    def forward(self, query, key, value, inputs_mask=None):
        """
        :param query: [query_size, max_len, embed_size]
        :param key: [key_size, max_len, embed_size]
        :param value: [key_size, max_len, embed_size]
        :param inputs_mask: [N, 1, max_len, max_len]
        :return: [N, max_len, embed_size]
        """
        batch_size = query.size(0)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(query.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        K = K.view(key.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        V = V.view(value.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if inputs_mask is not None:
            energy = energy.masked_fill(inputs_mask == 0, -10000000000.0)
        attention_prob = F.softmax(energy, dim=-1)
        attention_prob = self.attn_dropout(attention_prob)
        out = torch.matmul(attention_prob, V)
        out = out.permute((0, 2, 1, 3)).contiguous()
        out = out.view((batch_size, -1, self.embed_size))
        out = self.fc_o(out)
        return out, attention_prob


class PointWiseFeedForwardLayer(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super(PointWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, inputs):
        out = self.fc2(F.gelu(self.fc1(inputs)))
        return out


class EncoderLayer(nn.Module):

    def __init__(self, embed_size, ffn_hidden, num_heads, attn_dropout, hidden_dropout, layer_norm_eps):
        super(EncoderLayer, self).__init__()
        self.attn_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)
        self.pff_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)
        self.self_attention = MultiHeadAttentionLayer(embed_size, num_heads, attn_dropout)
        self.pff = PointWiseFeedForwardLayer(embed_size, ffn_hidden)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.pff_out_drop = nn.Dropout(hidden_dropout)

    def forward(self, input_seq, inputs_mask):
        """
        input:
            inputs: torch.FloatTensor, [batch_size, max_len, embed_size]
            inputs_mask: torch.BoolTensor, [batch_size, 1, 1, max_len]
        return:
            out_seq_embed: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        out_seq, att_matrix = self.self_attention(input_seq, input_seq, input_seq, inputs_mask)
        input_seq = self.attn_layer_norm(input_seq + self.hidden_dropout(out_seq))
        out_seq = self.pff(input_seq)
        out_seq = self.pff_layer_norm(input_seq + self.pff_out_drop(out_seq))
        return out_seq


class Transformer(nn.Module):

    def __init__(self, embed_size, ffn_hidden, num_blocks, num_heads, attn_dropout, hidden_dropout, layer_norm_eps=0.02, bidirectional=False):
        super(Transformer, self).__init__()
        self.bidirectional = bidirectional
        encoder_layer = EncoderLayer(embed_size=embed_size, ffn_hidden=ffn_hidden, num_heads=num_heads, attn_dropout=attn_dropout, hidden_dropout=hidden_dropout, layer_norm_eps=layer_norm_eps)
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_blocks)])

    def forward(self, item_input, seq_embedding):
        """
        Only output the sequence representations of the last layer in Transformer.
        out_seq_embed: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        mask = self.create_mask(item_input)
        for layer in self.encoder_layers:
            seq_embedding = layer(seq_embedding, mask)
        return seq_embedding

    def create_mask(self, input_seq):
        """
        Parameters:
            input_seq: torch.LongTensor, [batch_size, max_len]
        Return:
            mask: torch.BoolTensor, [batch_size, 1, max_len, max_len]
        """
        mask = (input_seq != 0).bool().unsqueeze(1).unsqueeze(2)
        mask = mask.expand(-1, -1, mask.size(-1), -1)
        if not self.bidirectional:
            mask = torch.tril(mask)
        return mask

    def set_attention_direction(self, bidirection=False):
        self.bidirectional = bidirection


class IOCRec(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(IOCRec, self).__init__(enc_dict, config)
        self.initializer_range = config.get('initializer_range', 0.02)
        self.aug_views = 2
        self.tao = config.get('tao', 2)
        self.all_hidden = config.get('all_hidden', True)
        self.lamda = config.get('lamda', 0.1)
        self.k_intention = config.get('K', 4)
        self.layer_norm_eps = self.config.get('layer_norm_eps', 1e-12)
        self.hidden_dropout = self.config.get('hidden_dropout', 0.5)
        self.ffn_hidden = self.config.get('ffn_hidden', 128)
        self.num_blocks = self.config.get('num_blocks', 3)
        self.num_heads = self.config.get('num_heads', 2)
        self.attn_dropout = self.config.get('attn_dropout', 0.5)
        self.position_embedding = nn.Embedding(self.max_length, self.embedding_dim)
        self.input_layer_norm = nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)
        self.input_dropout = nn.Dropout(self.hidden_dropout)
        self.local_encoder = Transformer(embed_size=self.embedding_dim, ffn_hidden=self.ffn_hidden, num_blocks=self.num_blocks, num_heads=self.num_heads, attn_dropout=self.attn_dropout, hidden_dropout=self.hidden_dropout, layer_norm_eps=self.layer_norm_eps)
        self.global_seq_encoder = GlobalSeqEncoder(embed_size=self.embedding_dim, max_len=self.max_length, dropout=self.hidden_dropout)
        self.disentangle_encoder = DisentangleEncoder(k_intention=self.k_intention, embed_size=self.embedding_dim, max_len=self.max_length)
        self.data_augmenter = DataAugmenter(num_items=self.enc_dict[self.config['item_col']]['vocab_size'] - 1)
        self.nce_loss = InfoNCELoss(temperature=self.tao, similarity_type='dot')
        self.cross_entropy = nn.CrossEntropyLoss()
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        seq_len = torch.sum(mask, dim=-1)
        local_seq_emb = self.local_seq_encoding(item_seq, seq_len, return_all=True)
        global_seq_emb = self.global_seq_encoding(item_seq, seq_len)
        disentangled_intention_emb = self.disentangle_encoder(local_seq_emb, global_seq_emb, seq_len)
        gather_index = seq_len.view(-1, 1, 1, 1).repeat(1, self.k_intention, 1, self.embedding_dim)
        user_emb = disentangled_intention_emb.gather(2, gather_index - 1).squeeze()
        if is_training:
            item = data['target_item'].squeeze()
            candidates = self.item_emb.weight.unsqueeze(0)
            logits = user_emb @ candidates.permute(0, 2, 1)
            max_logits, _ = torch.max(logits, 1)
            rec_loss = self.cross_entropy(max_logits, item)
            B = item.shape[0]
            aug_seq_1 = self.data_augmenter.augment(item_seq)
            aug_seq_2 = self.data_augmenter.augment(item_seq)
            aug_local_emb_1 = self.local_seq_encoding(aug_seq_1, seq_len, return_all=self.all_hidden)
            aug_global_emb_1 = self.global_seq_encoding(aug_seq_1, seq_len)
            disentangled_intention_1 = self.disentangle_encoder(aug_local_emb_1, aug_global_emb_1, seq_len)
            disentangled_intention_1 = disentangled_intention_1.view(B * self.k_intention, -1)
            aug_local_emb_2 = self.local_seq_encoding(aug_seq_2, seq_len, return_all=self.all_hidden)
            aug_global_emb_2 = self.global_seq_encoding(aug_seq_2, seq_len)
            disentangled_intention_2 = self.disentangle_encoder(aug_local_emb_2, aug_global_emb_2, seq_len)
            disentangled_intention_2 = disentangled_intention_2.view(B * self.k_intention, -1)
            cl_loss = self.nce_loss(disentangled_intention_1, disentangled_intention_2)
            loss = rec_loss + self.lamda * cl_loss
            output_dict = {'loss': loss}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict

    def position_encoding(self, item_input):
        seq_embedding = self.item_emb(item_input)
        position = torch.arange(self.max_length, device=item_input.device).unsqueeze(0)
        position = position.expand_as(item_input).long()
        pos_embedding = self.position_embedding(position)
        seq_embedding += pos_embedding
        seq_embedding = self.input_layer_norm(seq_embedding)
        seq_embedding = self.input_dropout(seq_embedding)
        return seq_embedding

    def local_seq_encoding(self, item_seq, seq_len, return_all=False):
        seq_embedding = self.position_encoding(item_seq)
        out_seq_embedding = self.local_encoder(item_seq, seq_embedding)
        if not return_all:
            out_seq_embedding = self.gather_indexes(out_seq_embedding, seq_len - 1)
        return out_seq_embedding

    def global_seq_encoding(self, item_seq, seq_len):
        return self.global_seq_encoder(item_seq, seq_len, self.item_emb)


class MIND(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(MIND, self).__init__(enc_dict, config)
        self.capsule = CapsuleNetwork(self.embedding_dim, self.max_length, bilinear_type=0, interest_num=self.config['K'])
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        if is_training:
            item = data['target_item'].squeeze()
            seq_emb = self.item_emb(item_seq)
            item_e = self.item_emb(item).squeeze(1)
            multi_interest_emb = self.capsule(seq_emb, mask, self.device)
            cos_res = torch.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            k_index = torch.argmax(cos_res, dim=1)
            best_interest_emb = torch.rand(multi_interest_emb.shape[0], multi_interest_emb.shape[2])
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]
            loss = self.calculate_loss(best_interest_emb, item)
            output_dict = {'user_emb': multi_interest_emb, 'loss': loss}
        else:
            seq_emb = self.item_emb(item_seq)
            multi_interest_emb = self.capsule(seq_emb, mask, self.device)
            output_dict = {'user_emb': multi_interest_emb}
        return output_dict


class NARM(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(NARM, self).__init__(enc_dict, config)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout_probs = self.config.get('dropout_probs', [0.1, 0.1])
        self.hidden_size = self.config.get('hidden_size', 32)
        self.emb_dropout = nn.Dropout(self.dropout_probs[0])
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers, bias=False, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs[1])
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_dim, bias=False)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)
        item_seq_emb = self.item_emb(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_out, _ = self.gru(item_seq_emb_dropout)
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        user_emb = self.b(c_t)
        if is_training:
            target_item = data['target_item'].squeeze()
            output_dict = {'user_emb': user_emb, 'loss': self.calculate_loss(user_emb, target_item)}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


class NextItNet(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(NextItNet, self).__init__(enc_dict, config)
        self.dilations = self.config.get('dilations', None)
        self.one_masked = self.config.get('one_masked', False)
        self.kernel_size = self.config.get('kernel_size', 3)
        self.feat_drop = self.config.get('feat_drop', 0)
        self.nextit_layer = NextItNetLayer(self.embedding_dim, self.dilations, self.one_masked, self.kernel_size, feat_drop=self.feat_drop)
        self.fc = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        item_seq_len = torch.sum(mask, dim=1)
        item_seq_emb = self.item_emb(item_seq)
        user_emb = self.nextit_layer(item_seq_emb, item_seq_len)
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            output_dict = {'user_emb': user_emb, 'loss': loss}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


class NISER(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(NISER, self).__init__(enc_dict, config)
        self.step = self.config.get('step', 1)
        self.pos_embedding = nn.Embedding(self.max_length, self.embedding_dim)
        self.item_dropout = nn.Dropout(config.get('item_dropout', 0.1))
        self.gnncell = SRGNNCell(self.embedding_dim)
        self.linear_one = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_two = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_three = nn.Linear(self.embedding_dim, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """
        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)
        batch_data = generate_graph(data)
        hidden = self.item_emb(batch_data['x'])
        hidden = self.item_dropout(hidden)
        hidden = F.normalize(hidden, dim=-1)
        for i in range(self.step):
            hidden = self.gnncell(batch_data['in_graph'], batch_data['out_graph'], hidden)
        seq_hidden = hidden[batch_data['alias_inputs']]
        pos_emb = self.pos_embedding.weight[:seq_hidden.shape[1]]
        pos_emb = pos_emb.unsqueeze(0).expand(item_seq_len.shape[0], -1, -1)
        seq_hidden = seq_hidden + pos_emb
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        """
        在使用attention score进行加权求和的时候，我们对无效位置需要进行mask，即直接乘以mask即可
        """
        a = torch.sum(alpha * seq_hidden * data['hist_mask_list'].view(seq_hidden.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        seq_output = F.normalize(seq_output, dim=-1)
        output_dict = dict()
        if is_training:
            loss = self.calculate_loss(seq_output, data['target_item'].squeeze())
            output_dict['loss'] = loss
        else:
            output_dict['user_emb'] = seq_output
        return output_dict


class Re4(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(Re4, self).__init__(enc_dict, config)
        self.num_interests = self.config.get('K', 4)
        self.att_thre = self.config.get('att_thre', -1)
        self.t_cont = self.config.get('t_cont', 0.02)
        self.att_lambda = self.config.get('att_lambda', 0.01)
        self.ct_lambda = self.config.get('ct_lambda', 0.1)
        self.cs_lambda = self.config.get('cs_lambda', 0.1)
        self.proposal_num = self.num_interests
        self.W1 = torch.nn.Parameter(data=torch.randn(256, self.embedding_dim), requires_grad=True)
        self.W1_2 = torch.nn.Parameter(data=torch.randn(self.proposal_num, 256), requires_grad=True)
        self.W2 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.W3 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.W3_2 = torch.nn.Parameter(data=torch.randn(self.max_length, self.embedding_dim), requires_grad=True)
        self.W5 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc_cons = nn.Linear(self.embedding_dim, self.embedding_dim * self.max_length)
        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        self.recons_mse_loss = nn.MSELoss(reduce=False)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        item_mask = 1 - data['hist_mask_list']
        dim0, dim1 = item_seq.shape
        item_seq_len = torch.sum(item_mask, dim=-1)
        item_seq = torch.reshape(item_seq, (1, dim0 * dim1))
        item_seq_emb = self.item_emb(item_seq)
        item_seq_emb = torch.reshape(item_seq_emb, (dim0, dim1, -1))
        proposals_weight = torch.matmul(self.W1_2, torch.tanh(torch.matmul(self.W1, torch.transpose(item_seq_emb, 1, 2))))
        proposals_weight_logits = proposals_weight.masked_fill(item_mask.unsqueeze(1).bool(), -1000000000.0)
        proposals_weight = torch.softmax(proposals_weight_logits, dim=2)
        user_interests = torch.matmul(proposals_weight, torch.matmul(item_seq_emb, self.W2))
        if is_training:
            target_item = data['target_item']
            item_e = self.item_emb(target_item)
            product = torch.matmul(user_interests, torch.transpose(item_seq_emb, 1, 2))
            product = product.masked_fill(item_mask.unsqueeze(1).bool(), -1000000000.0)
            re_att = torch.softmax(product, dim=2)
            att_pred = F.log_softmax(proposals_weight_logits, dim=-1)
            loss_attend = -(re_att * att_pred).sum() / re_att.sum()
            norm_watch_interests = F.normalize(user_interests, p=2, dim=-1)
            norm_watch_movie_embedding = F.normalize(item_seq_emb, p=2, dim=-1)
            cos_sim = torch.matmul(norm_watch_interests, torch.transpose(norm_watch_movie_embedding, 1, 2))
            if self.att_thre == -1:
                gate = np.repeat(1 / (item_seq_len.cpu() * 1.0), self.max_length, axis=0)
            else:
                gate = np.repeat(torch.FloatTensor([self.att_thre]).repeat(item_seq_len.size(0)), self.max_length, axis=0)
            gate = torch.reshape(gate, (dim0, 1, self.max_length))
            positive_weight_idx = (proposals_weight > gate) * 1
            mask_cos = cos_sim.masked_fill(item_mask.unsqueeze(1).bool(), -1000000000.0)
            pos_cos = mask_cos.masked_fill(positive_weight_idx != 1, -1000000000.0)
            cons_pos = torch.exp(pos_cos / self.t_cont)
            cons_neg = torch.sum(torch.exp(mask_cos / self.t_cont), dim=2)
            in2in = torch.matmul(norm_watch_interests, torch.transpose(norm_watch_interests, 1, 2))
            in2in = in2in.masked_fill(torch.eye(self.proposal_num).unsqueeze(0) == 1, -1000000000.0)
            cons_neg = cons_neg + torch.sum(torch.exp(in2in / self.t_cont), dim=2)
            item_rolled = torch.roll(norm_watch_movie_embedding, 1, 0)
            in2i = torch.matmul(norm_watch_interests, torch.transpose(item_rolled, 1, 2))
            in2i_mask = torch.roll((item_seq == 0).reshape(dim0, dim1), 1, 0)
            in2i = in2i.masked_fill(in2i_mask.unsqueeze(1), -1000000000.0)
            cons_neg = cons_neg + torch.sum(torch.exp(in2i / self.t_cont), dim=2)
            cons_div = cons_pos / cons_neg.unsqueeze(-1)
            cons_div = cons_div.masked_fill(item_mask.unsqueeze(1).bool(), 1)
            cons_div = cons_div.masked_fill(positive_weight_idx != 1, 1)
            loss_contrastive = -torch.log(cons_div)
            loss_contrastive = torch.mean(loss_contrastive)
            recons_item = self.fc_cons(user_interests)
            recons_item = recons_item.reshape([dim0 * self.proposal_num, dim1, -1])
            recons_weight = torch.matmul(self.W3_2, torch.tanh(torch.matmul(self.W3, torch.transpose(recons_item, 1, 2))))
            recons_weight = recons_weight.reshape([dim0, self.proposal_num, dim1, dim1])
            recons_weight = recons_weight.masked_fill((item_seq == 0).reshape(dim0, 1, 1, dim1), -1000000000.0).reshape([-1, dim1, dim1])
            recons_weight = torch.softmax(recons_weight, dim=-1)
            recons_item = torch.matmul(recons_weight, torch.matmul(recons_item, self.W5)).reshape([dim0, self.proposal_num, dim1, -1])
            target_emb = item_seq_emb.unsqueeze(1).repeat(1, self.proposal_num, 1, 1)
            loss_construct = self.recons_mse_loss(recons_item, target_emb)
            loss_construct = loss_construct.masked_fill((positive_weight_idx == 0).unsqueeze(-1), 0.0)
            loss_construct = loss_construct.masked_fill(item_mask.unsqueeze(-1).unsqueeze(1).bool(), 0.0)
            loss_construct = torch.mean(loss_construct)
            user_interests = F.tanh(self.fc1(user_interests))
            cos_res = torch.bmm(user_interests, item_e.squeeze(1).unsqueeze(-1))
            k_index = torch.argmax(cos_res, dim=1)
            best_interest_emb = torch.rand(user_interests.shape[0], user_interests.shape[2])
            for k in range(user_interests.shape[0]):
                best_interest_emb[k, :] = user_interests[k, k_index[k], :]
            loss = self.calculate_loss(best_interest_emb, target_item.squeeze())
            loss = loss + self.att_lambda * loss_attend + self.ct_lambda * loss_contrastive + self.cs_lambda * loss_construct
            output_dict = {'loss': loss}
        else:
            user_interests = F.tanh(self.fc1(user_interests))
            output_dict = {'user_emb': user_interests}
        return output_dict


class SASRec(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(SASRec, self).__init__(enc_dict, config)
        self.n_layers = config.get('n_layers', 2)
        self.n_heads = config.get('n_heads', 4)
        self.hidden_size = config.get('hidden_size', 64)
        self.inner_size = config.get('inner_size', 32)
        self.hidden_dropout_prob = config.get('hidden_dropout_prob', 0.1)
        self.attn_dropout_prob = config.get('attn_dropout_prob', 0.1)
        self.hidden_act = config.get('hidden_act', 'gelu')
        self.layer_norm_eps = config.get('layer_norm_eps', 0.001)
        self.self_attention = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads, hidden_size=self.embedding_dim, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob, attn_dropout_prob=self.attn_dropout_prob, hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """
        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)
        item_seq_emb = self.item_emb(data['hist_item_list'])
        attention_mask = self.get_attention_mask(data['hist_mask_list'])
        outputs = self.self_attention(item_seq_emb, attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        user_emb = self.gather_indexes(output, item_seq_len - 1)
        if is_training:
            target_item = data['target_item'].squeeze()
            output_dict = {'user_emb': user_emb, 'loss': self.calculate_loss(user_emb, target_item)}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


class SINE(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(SINE, self).__init__(enc_dict, config)
        self.layer_norm_eps = self.config.get('layer_norm_eps', 0.0001)
        self.D = self.embedding_dim
        self.L = self.config.get('prototype_size', 500)
        self.k = self.config.get('interest_size', 4)
        self.tau = self.config.get('tau_ratio', 0.1)
        self.reg_loss_ratio = self.config.get('reg_loss_ratio', 0.1)
        self.initializer_range = 0.01
        self.w1 = self._init_weight((self.D, self.D))
        self.w2 = self._init_weight(self.D)
        self.w3 = self._init_weight((self.D, self.D))
        self.w4 = self._init_weight(self.D)
        self.C = nn.Embedding(self.L, self.D)
        self.w_k_1 = self._init_weight((self.k, self.D, self.D))
        self.w_k_2 = self._init_weight((self.k, self.D))
        self.ln2 = nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)
        self.ln4 = nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)
        self.reset_parameters()

    def _init_weight(self, shape):
        mat = torch.FloatTensor(np.random.normal(0, self.initializer_range, shape))
        return nn.Parameter(mat, requires_grad=True)

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        x_u = self.item_emb(item_seq)
        x = torch.matmul(x_u, self.w1)
        x = torch.tanh(x)
        x = torch.matmul(x, self.w2)
        a = F.softmax(x, dim=1)
        z_u = torch.matmul(a.unsqueeze(2).transpose(1, 2), x_u).transpose(1, 2)
        s_u = torch.matmul(self.C.weight, z_u)
        s_u = s_u.squeeze(2)
        idx = s_u.argsort(1)[:, -self.k:]
        s_u_idx = s_u.sort(1)[0][:, -self.k:]
        c_u = self.C(idx)
        sigs = torch.sigmoid(s_u_idx.unsqueeze(2).repeat(1, 1, self.embedding_dim))
        C_u = c_u.mul(sigs)
        w3_x_u_norm = F.normalize(x_u.matmul(self.w3), p=2, dim=2)
        C_u_norm = self.ln2(C_u)
        P_k_t = torch.bmm(w3_x_u_norm, C_u_norm.transpose(1, 2))
        P_k_t_b = F.softmax(P_k_t, dim=2)
        P_k_t_b_t = P_k_t_b.transpose(1, 2)
        a_k = x_u.unsqueeze(1).repeat(1, self.k, 1, 1).matmul(self.w_k_1)
        P_t_k = F.softmax(torch.tanh(a_k).matmul(self.w_k_2.reshape(self.k, self.embedding_dim, 1)).squeeze(3), dim=2)
        mul_p = P_k_t_b_t.mul(P_t_k)
        x_u_re = x_u.unsqueeze(1).repeat(1, self.k, 1, 1)
        mul_p_re = mul_p.unsqueeze(3)
        delta_k = x_u_re.mul(mul_p_re).sum(2)
        delta_k = F.normalize(delta_k, p=2, dim=2)
        x_u_bar = P_k_t_b.matmul(C_u)
        C_apt = F.softmax(torch.tanh(x_u_bar.matmul(self.w3)).matmul(self.w4), dim=1)
        C_apt = C_apt.reshape(-1, 1, self.max_length).matmul(x_u_bar)
        C_apt = self.ln4(C_apt)
        e_k = delta_k.bmm(C_apt.reshape(-1, self.embedding_dim, 1)) / self.tau
        e_k_u = F.softmax(e_k.squeeze(2), dim=1)
        user_emb = e_k_u.unsqueeze(2).mul(delta_k).sum(dim=1)
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            output_dict = {'user_emb': user_emb, 'loss': loss}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


class SRGNN(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(SRGNN, self).__init__(enc_dict, config)
        self.step = self.config.get('step', 1)
        self.gnncell = SRGNNCell(self.embedding_dim)
        self.linear_one = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_two = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_three = nn.Linear(self.embedding_dim, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq_len = torch.sum(data['hist_mask_list'], dim=-1)
        batch_data = generate_graph(data)
        hidden = self.item_emb(batch_data['x'])
        for i in range(self.step):
            hidden = self.gnncell(batch_data['in_graph'], batch_data['out_graph'], hidden)
        seq_hidden = hidden[batch_data['alias_inputs']]
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        """
        在使用attention score进行加权求和的时候，我们对无效位置需要进行mask，即直接乘以mask即可
        """
        a = torch.sum(alpha * seq_hidden * data['hist_mask_list'].view(seq_hidden.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        output_dict = dict()
        if is_training:
            loss = self.calculate_loss(seq_output, data['target_item'].squeeze())
            output_dict['loss'] = loss
        else:
            output_dict['user_emb'] = seq_output
        return output_dict


class STAMP(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(STAMP, self).__init__(enc_dict, config)
        self.feat_drop = self.config.get('feat_drop', 0)
        self.stamp_layer = STAMPLayer(self.embedding_dim, feat_drop=self.feat_drop)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        item_seq_len = torch.sum(mask, dim=1)
        item_seq_emb = self.item_emb(item_seq)
        user_emb = self.stamp_layer(item_seq_emb, item_seq_len)
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            output_dict = {'user_emb': user_emb, 'loss': loss}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


class YotubeDNN(SequenceBaseModel):

    def __init__(self, enc_dict, config):
        super(YotubeDNN, self).__init__(enc_dict, config)
        self.reset_parameters()

    def forward(self, data: 'Dict[str, torch.tensor]', is_training: 'bool'=True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for 
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding 
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        user_emb = self.item_emb(item_seq)
        mask = mask.unsqueeze(-1).float()
        user_emb = torch.mean(user_emb * mask, dim=1)
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            output_dict = {'user_emb': user_emb, 'loss': loss}
        else:
            output_dict = {'user_emb': user_emb}
        return output_dict


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BilinearInteractionLayer,
     lambda: ([], {'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CCPM_ConvLayer,
     lambda: ([], {'num_fields': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     True),
    (CompressedInteractionNet,
     lambda: ([], {'num_fields': 4, 'cin_layer_units': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ContraLoss,
     lambda: ([], {'device': 0}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CrossInteractionLayer,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossNet,
     lambda: ([], {'input_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dice,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (EncoderLayer,
     lambda: ([], {'embed_size': 4, 'ffn_hidden': 4, 'num_heads': 4, 'attn_dropout': 0.5, 'hidden_dropout': 0.5, 'layer_norm_eps': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FM_Layer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GeneralizedInteraction,
     lambda: ([], {'input_subspaces': 4, 'output_subspaces': 4, 'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (GeneralizedInteractionNet,
     lambda: ([], {'num_layers': 1, 'num_subspaces': 4, 'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GraphLayer,
     lambda: ([], {'num_fields': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (InnerProductLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InteractionMachine,
     lambda: ([], {'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KMaxPooling,
     lambda: ([], {'k': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskBlock,
     lambda: ([], {'input_dim': 4, 'mask_input_dim': 4, 'output_size': 4, 'reduction_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaskedAveragePooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MaskedSumPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'n_heads': 4, 'hidden_size': 4, 'hidden_dropout_prob': 0.5, 'attn_dropout_prob': 0.5, 'layer_norm_eps': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MultiHeadAttentionLayer,
     lambda: ([], {'embed_size': 4, 'nhead': 4, 'attn_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiInterestSelfAttention,
     lambda: ([], {'embedding_dim': 4, 'num_attention_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (PointWiseFeedForwardLayer,
     lambda: ([], {'embed_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlockOneMasked,
     lambda: ([], {'channels': 4, 'kernel_size': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ResBlockTwoMasked,
     lambda: ([], {'channels': 4, 'kernel_size': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (SENET_Layer,
     lambda: ([], {'num_fields': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SqueezeExcitationLayer,
     lambda: ([], {'num_fields': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_HaSai666_rec_pangu(_paritybench_base):
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

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

