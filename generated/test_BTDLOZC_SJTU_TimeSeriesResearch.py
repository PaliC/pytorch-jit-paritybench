import sys
_module = sys.modules[__name__]
del sys
data = _module
data_loader = _module
exp = _module
exp_DeepAR = _module
exp_DeepFactor = _module
exp_LSTNet = _module
exp_MLP = _module
exp_MLP_proba = _module
exp_TCN = _module
exp_Transformer = _module
exp_basic = _module
DeepAR_network = _module
DeepAR = _module
DeepFactor_network = _module
DeepFactor = _module
LSTNet_network = _module
LSTNet = _module
MLP_network = _module
MLP = _module
MLP_proba_network = _module
MLP_proba = _module
TCN_network = _module
TCN = _module
Transformer_network = _module
Transformer = _module
attn = _module
models = _module
modules = _module
distribution_output = _module
lambda_layer = _module
test = _module
data_test = _module
data_loader_test = _module
DeepAR_test = _module
DeepFactor_test = _module
LSTNet_test = _module
MLP_proba_test = _module
MLP_test = _module
TCN_test = _module
models_test = _module
utils = _module
embed = _module
metrics = _module
metrics_proba = _module
plot_proba_forcast = _module
scaler = _module
time_features = _module
time_lags = _module
tools = _module

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


import numpy as np


import pandas as pd


from typing import Optional


from typing import List


from typing import Tuple


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import warnings


import matplotlib.pyplot as plt


import math


import torch.nn as nn


from torch.distributions.normal import Normal


from torch import optim


from torch.nn import RNN


from torch.nn import LSTM


from torch.nn import GRU


from torch import Tensor


from pandas.tseries.frequencies import to_offset


from typing import Callable


import torch.nn.functional as F


from torch.nn.utils import weight_norm


from torch.nn.init import xavier_uniform_


from torch.nn.init import xavier_normal_


from torch.nn.init import constant_


from typing import Dict


from torch.distributions import AffineTransform


from torch.distributions import Beta


from torch.distributions import Distribution


from torch.distributions import Gamma


from torch.distributions import NegativeBinomial


from torch.distributions import Normal


from torch.distributions import Poisson


from torch.distributions import StudentT


from torch.distributions import TransformedDistribution


from functools import reduce


from pandas.tseries import offsets


class LambdaLayer(nn.Module):

    def __init__(self, function):
        super().__init__()
        self._func = function

    def forward(self, x, *args):
        return self._func(x, *args)


class PtArgProj(nn.Module):
    """
    A PyTorch module that can be used to project from a dense layer
    to PyTorch distribution arguments.
    Parameters
    ----------
    in_features
        Size of the incoming features.
    dim_args
        Dictionary with string key and int value
        dimension of each arguments that will be passed to the domain
        map, the names are not used.
    domain_map
        Function returning a tuple containing one tensor
        a function or a nn.Module. This will be called with num_args
        arguments and should return a tuple of outputs that will be
        used when calling the distribution constructor.
    """

    def __init__(self, in_features: 'int', args_dim: 'Dict[str, int]', domain_map: 'Callable[..., Tuple[torch.Tensor]]', **kwargs) ->None:
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.proj = nn.ModuleList([nn.Linear(in_features, dim) for dim in args_dim.values()])
        self.domain_map = domain_map

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]
        return self.domain_map(*params_unbounded)


class Output:
    """
    Class to connect a network to some output
    """
    in_features: 'int'
    args_dim: 'Dict[str, int]'
    _dtype: 'np.float32'

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    def get_args_proj(self, in_features: 'int') ->nn.Module:
        return PtArgProj(in_features=in_features, args_dim=self.args_dim, domain_map=LambdaLayer(self.domain_map))

    def domain_map(self, *args: torch.Tensor):
        raise NotImplementedError()


class DistributionOutput(Output):
    """
    Class to construct a distribution given the output of a network.
    """
    distr_cls: 'type'

    def __init__(self) ->None:
        pass

    def distribution(self, distr_args, loc: 'Optional[torch.Tensor]'=None, scale: 'Optional[torch.Tensor]'=None) ->Distribution:
        """
        Construct the associated distribution, given the collection of
        constructor arguments and, optionally, a scale tensor.
        Parameters
        ----------
        distr_args
            Constructor arguments for the underlying Distribution type.
        loc
            Optional tensor, of the same shape as the
            batch_shape+event_shape of the resulting distribution.
        scale
            Optional tensor, of the same shape as the
            batch_shape+event_shape of the resulting distribution.
        """
        if loc is None and scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedDistribution(distr, [AffineTransform(loc=0.0 if loc is None else loc, scale=1.0 if scale is None else scale)])

    @property
    def event_shape(self) ->Tuple:
        """
        Shape of each individual event contemplated by the distributions
        that this object constructs.
        """
        raise NotImplementedError()

    @property
    def event_dim(self) ->int:
        """
        Number of event dimensions, i.e., length of the `event_shape` tuple,
        of the distributions that this object constructs.
        """
        return len(self.event_shape)

    @property
    def value_in_support(self) ->float:
        """
        A float that will have a valid numeric value when computing the
        log-loss of the corresponding distribution. By default 0.0.
        This value will be used when padding data series.
        """
        return 0.0

    def domain_map(self, *args: torch.Tensor):
        """
        Converts arguments to the right shape and domain. The domain depends
        on the type of distribution, while the correct shape is obtained by
        reshaping the trailing axis in such a way that the returned tensors
        define a distribution of the right event_shape.
        """
        raise NotImplementedError()


class StudentTOutput(DistributionOutput):
    args_dim: 'Dict[str, int]' = {'df': 1, 'loc': 1, 'scale': 1}
    distr_cls: 'type' = StudentT

    @classmethod
    def domain_map(cls, df: 'torch.Tensor', loc: 'torch.Tensor', scale: 'torch.Tensor'):
        scale = F.softplus(scale)
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) ->Tuple:
        return ()


def get_lagged_subsequences_by_default(sequence: 'torch.Tensor', sequence_len: 'int', subsequence_len: 'int', mode: 'bool'):
    lagged_values = []
    if mode == True:
        for i in range(1, sequence_len - subsequence_len + 1):
            begin_index = -i - subsequence_len
            end_index = -i
            lagged_values.append(sequence[:, begin_index:end_index, ...])
    else:
        for i in range(0, sequence_len - subsequence_len):
            begin_index = -i - subsequence_len
            end_index = -i if i > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
    return torch.stack(lagged_values, dim=-1)


def mean_abs_scaling(context, min_scale=1e-05):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class DeepAR(nn.Module):

    def __init__(self, c_in: 'int', c_out: 'int', d_model: 'int', hist_len: 'int', cntx_len: 'int', pred_len: 'int', num_layers: 'int'=1, hidden_size: 'int'=40, embedding_dim: 'int'=10, cell_type: 'str'='GRU', dropout_rate: 'float'=0.0, freq: 'str'='H', num_parallel_samples: 'int'=100, distr_output: 'Callable'=StudentTOutput(), scaling: 'Callable'=mean_abs_scaling):
        super(DeepAR, self).__init__()
        assert c_in == c_out, 'Auto-regressive model should have same input dimension and output dimension'
        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'B': 4, 'H': 5, 'T': 6, 'S': 7}
        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.hist_len = hist_len
        self.cntx_len = cntx_len
        self.pred_len = pred_len
        self.freq = to_offset(freq)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim * freq_map[self.freq.name]
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.num_parallel_samples = num_parallel_samples
        self.distr_output = distr_output
        self.scaling = scaling
        self.time_feat_embedding = nn.Linear(freq_map[self.freq.name], self.embedding_dim)
        rnn_cell_map = {'RNN': RNN, 'LSTM': LSTM, 'GRU': GRU}
        self.rnn = rnn_cell_map[cell_type](input_size=self.embedding_dim + self.hist_len - self.cntx_len, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.args_proj = self.distr_output.get_args_proj(hidden_size)

    def unroll_encoder(self, x: 'Tensor', x_mark: 'Tensor', y: 'Optional[Tensor]', y_mark: 'Optional[Tensor]'):
        if y is None or y_mark is None:
            time_feat = x_mark[:, self.hist_len - self.cntx_len:]
            sequence = x
            sequence_len = self.hist_len
            subsequence_len = self.cntx_len
        else:
            time_feat = torch.cat((x_mark[:, self.hist_len - self.cntx_len:], y_mark), dim=1)
            sequence = torch.cat((x, y), dim=1)
            sequence_len = self.hist_len + self.pred_len
            subsequence_len = self.cntx_len + self.pred_len
        lags = get_lagged_subsequences_by_default(sequence, sequence_len, subsequence_len, True)
        time_feat = self.time_feat_embedding(time_feat)
        scale = self.scaling(x[:, -self.cntx_len:, :])
        sequence = sequence / scale
        lags_scale = lags / scale.unsqueeze(-1)
        input_lags = lags_scale.reshape(sequence.shape[0], subsequence_len, -1)
        inputs = torch.cat((input_lags, time_feat[:, -subsequence_len:, ...]), dim=-1)
        outputs, state = self.rnn(inputs)
        return outputs, state, scale

    def sampling_decoder(self, x: 'Tensor', x_mark: 'Tensor', y_mark: 'Tensor', begin_state: 'Tensor', scale: 'Tensor'):
        repeated_x = x.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
        repeated_y_mark = y_mark.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
        repeated_y_mark = self.time_feat_embedding(repeated_y_mark)
        repeated_scale = scale.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
        if self.cell_type == 'LSTM':
            repeated_states = [s.repeat_interleave(repeats=self.num_parallel_samples, dim=1) for s in begin_state]
        else:
            repeated_states = begin_state.repeat_interleave(repeats=self.num_parallel_samples, dim=1)
        future_samples = []
        for k in range(self.pred_len):
            lags = get_lagged_subsequences_by_default(repeated_x, self.hist_len - self.cntx_len + 1, 1, False)
            lags_scale = lags / scale.unsqueeze(-1)
            input_lags = lags_scale.reshape(repeated_y_mark.shape[0], 1, -1)
            decoder_inputs = torch.cat((input_lags, repeated_y_mark[:, k, :].unsqueeze(1)), dim=-1)
            rnn_outputs, repeated_states = self.rnn(decoder_inputs, repeated_states)
            distr_args = self.args_proj(rnn_outputs.unsqueeze(2))
            distr = self.distr_output.distribution(distr_args, scale=repeated_scale)
            new_samples = distr.sample()
            repeated_x = torch.cat((repeated_x, new_samples), dim=1)
            future_samples.append(new_samples)
        samples = torch.cat(future_samples, dim=1)
        return samples.reshape((self.num_parallel_samples, self.pred_len, self.c_out))

    def forward(self, x: 'Tensor', x_mark: 'Tensor', y: 'Optional[Tensor]', y_mark: 'Optional[Tensor]', mode: 'bool'):
        if mode:
            rnn_outputs, _, scale = self.unroll_encoder(x, x_mark, y, y_mark)
            distr_args = self.args_proj(rnn_outputs.unsqueeze(2))
            return self.distr_output.distribution(distr_args, scale=scale)
        else:
            _, state, scale = self.unroll_encoder(x, x_mark, y, y_mark)
            return self.sampling_decoder(x, x_mark, y_mark, state, scale)


class RecurrentModule(nn.Module):

    def __init__(self, cell_type: 'str', input_size: 'int', hidden_size: 'int', num_layers: 'int', dropout: 'float'=0.0, bidirectional: 'bool'=False):
        super(RecurrentModule, self).__init__()
        rnn_cell_map = {'RNN': RNN, 'LSTM': LSTM, 'GRU': GRU}
        self.rnn = rnn_cell_map[cell_type](input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        return self.rnn(x)


class DeepFactor(nn.Module):

    def __init__(self, c_in: 'int', c_out: 'int', d_model: 'int', hist_len: 'int', pred_len: 'int', num_hidden_global: 'int'=50, num_layers_global: 'int'=1, num_factors: 'int'=10, num_hidden_local: 'int'=5, num_layers_local: 'int'=1, embedding_dim: 'int'=10, cell_type: 'str'='GRU', freq: 'str'='H', use_time_feat: 'bool'=True):
        super(DeepFactor, self).__init__()
        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'B': 4, 'H': 5, 'T': 6, 'S': 7}
        self.c_in = c_in
        self.c_out = c_out
        self.freq = to_offset(freq)
        self.embedding_dim = embedding_dim * freq_map[self.freq.name]
        self.global_model = RecurrentModule(cell_type=cell_type.upper(), input_size=freq_map[self.freq.name], hidden_size=num_hidden_global, num_layers=num_layers_global, num_factors=num_factors, bidirectional=True)
        self.local_model = RecurrentModule(cell_type=cell_type.upper(), input_size=freq_map[self.freq.name] + self.embedding_dim, hidden_size=num_hidden_global, num_layers=num_layers_local, num_factors=1, bidirectional=True)
        self.assemble_features_embedding = nn.Linear(1, self.embedding_dim)
        self.loading = nn.Linear(self.embedding_dim, num_factors, bias=False)
        self.freq = freq

    def assemble_features(self, x_mark: 'Tensor'):
        latent_feat = torch.zeros(size=(x_mark.shape[0], 1))
        embed_feat = self.assemble_features_embedding(latent_feat)
        helper_ones = torch.ones(size=(x_mark.shape[0], x_mark.shape[1], 1))
        repeated_cat = torch.bmm(helper_ones, embed_feat.unsqueeze(1))
        local_input = torch.cat((repeated_cat, x_mark), dim=2)
        return embed_feat, local_input

    def forward(self, x, x_mark, y_mark):
        embed_feat, local_input = self.assemble_features(x_mark)
        loadings = self.loading(embed_feat)
        global_factors = self.global_model(x_mark)
        fixed_effect = torch.bmm(global_factors, loadings.unsqueeze(2))
        fixed_effect = torch.exp(fixed_effect)
        random_effect = torch.log(torch.exp(self.local_model(local_input)) + 1.0)
        return fixed_effect, random_effect


class LSTNet(nn.Module):

    def __init__(self, c_in: 'int', c_out: 'int', hist_len: 'int', pred_len: 'int', out_channels: 'int', kernel_size: 'int', cell_type: 'str', hidden_size: 'int', num_layers: 'int', skip_cell_type: 'str', skip_hidden_size: 'int', skip_num_layers: 'int', skip_size: 'int', ar_window: 'int', embedding_dim: 'int'=10, dropout_rate: 'float'=0.2, freq: 'str'='H'):
        """
        :param c_in:
        :param c_out:
        :param hist_len:
        :param pred_len:
        :param out_channels: Number of channels for first layer Conv2D
        :param kernel_size:
        :param cell_type:
        :param hidden_size:
        :param num_layers:
        :param skip_cell_type:
        :param skip_hidden_size:
        :param skip_num_layers:
        :param skip_size: Skip size for skip-RNN layers
        :param ar_window: Auto-regressive window size for the linear part
        :param embedding_dim:
        :param freq:
        """
        super(LSTNet, self).__init__()
        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'B': 4, 'H': 5, 'T': 6, 'S': 7}
        self.c_in = c_in
        self.c_out = c_out
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.skip_cell_type = skip_cell_type
        self.skip_hidden_size = skip_hidden_size
        self.skip_num_layers = skip_num_layers
        self.skip_size = skip_size
        self.skip_num = int((hist_len - kernel_size) / skip_size)
        self.ar_window = ar_window
        self.embedding_dim = embedding_dim
        self.freq = to_offset(freq)
        self.dropout_rate = dropout_rate
        self.time_feat_embedding = nn.Linear(freq_map[self.freq.name], self.embedding_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, c_in + embedding_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = RecurrentModule(cell_type=cell_type, input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate)
        self.skip_rnn = RecurrentModule(cell_type=skip_cell_type, input_size=out_channels, hidden_size=skip_hidden_size, num_layers=skip_num_layers, dropout=dropout_rate)
        self.projection = nn.Linear(hidden_size + skip_hidden_size * skip_size, pred_len * c_out)
        self.highway = nn.Linear(ar_window * (c_in + embedding_dim), pred_len * c_out)

    def forward(self, x, x_mark, y_mark):
        x_mark_emb = self.time_feat_embedding(x_mark)
        x = torch.cat((x, x_mark_emb), dim=-1)
        c = x.unsqueeze(1)
        c = self.conv(c)
        c = c.squeeze(3)
        c = self.relu(c)
        c = self.dropout(c)
        r = c.transpose(1, 2)
        _, r = self.rnn(r)
        r = r.squeeze(0)
        s = c[:, :, int(-self.skip_num * self.skip_size):].contiguous()
        s = s.view(x.shape[0], self.out_channels, self.skip_num, self.skip_size)
        s = s.permute(0, 3, 2, 1).contiguous()
        s = s.view(x.shape[0] * self.skip_size, self.skip_num, self.out_channels)
        _, s = self.skip_rnn(s)
        s = s.view(x.shape[0], self.skip_size * self.skip_hidden_size)
        r = torch.cat((r, s), dim=1)
        r = self.projection(r)
        r = r.view(x.shape[0], self.pred_len, self.c_out)
        z = x[:, -self.ar_window:, :]
        z = z.view(x.shape[0], self.ar_window * x.shape[2])
        z = self.highway(z)
        z = z.view(x.shape[0], self.pred_len, self.c_out)
        ret = r + z
        return ret


class TimeFeatureEmbedding(nn.Module):

    def __init__(self, d_model, freq='H'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'H': 5, 'T': 6, 'S': 7}
        d_inp = freq_map[freq.upper()]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class TokenLinearEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(TokenLinearEmbedding, self).__init__()
        self.tokenLinear = nn.Linear(c_in, d_model)

    def forward(self, x):
        x = self.tokenLinear(x)
        return x


class DataEmbedding(nn.Module):

    def __init__(self, c_in: 'int', d_model: 'int', freq='H', dropout=0.1, use_time_feat=True):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenLinearEmbedding(c_in, d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model, freq)
        self.dropout = nn.Dropout(p=dropout)
        self.use_time_feat = use_time_feat

    def forward(self, x, x_mark):
        if self.use_time_feat:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        else:
            x = self.value_embedding(x)
        return x


class MLP(nn.Module):

    def __init__(self, c_in: 'int', c_out: 'int', d_model: 'int', num_hidden_dimensions: 'List[int]', hist_len: 'int', pred_len: 'int', freq: 'str'='H', use_time_feat: 'bool'=True):
        super(MLP, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.num_hidden_dimensions = num_hidden_dimensions
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.freq = freq
        self.embedding = DataEmbedding(c_in, d_model, freq, use_time_feat=use_time_feat)
        modules = []
        dims = self.num_hidden_dimensions
        for i, units in enumerate(dims):
            if i == 0:
                input_size = hist_len
            else:
                input_size = dims[i - 1]
            modules += [nn.Linear(input_size, units), nn.ReLU()]
        modules.append(nn.Linear(dims[-1], pred_len))
        self.mlp = nn.Sequential(*modules)
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x, x_mark, y_mark):
        x = self.embedding(x, x_mark)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        return x


class MLP_proba(nn.Module):

    def __init__(self, c_in: 'int', c_out: 'int', d_model: 'int', num_hidden_dimensions: 'List[int]', hist_len: 'int', pred_len: 'int', freq: 'str'='H', use_time_feat: 'bool'=True, distr_output: 'Callable'=StudentTOutput(), scaling: 'Callable'=mean_abs_scaling):
        super(MLP_proba, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.num_hidden_dimensions = num_hidden_dimensions
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.freq = freq
        self.distr_output = distr_output
        self.scaling = scaling
        self.embedding = DataEmbedding(c_in, d_model, freq, use_time_feat=use_time_feat)
        modules = []
        dims = self.num_hidden_dimensions
        for i, units in enumerate(dims):
            if i == 0:
                input_size = hist_len
            else:
                input_size = dims[i - 1]
            modules += [nn.Linear(input_size, units), nn.ReLU()]
        modules.append(nn.Linear(dims[-1], pred_len * dims[-1]))
        self.projection = nn.Linear(d_model, c_out)
        self.mlp = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(dims[-1])

    def forward(self, x, x_mark, y_mark):
        x = self.embedding(x, x_mark)
        scale = self.scaling(x)
        x = x / scale
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], self.pred_len, x.shape[1], -1)
        distr_args = self.args_proj(x)
        distr = self.distr_output.distribution(distr_args, scale=scale)
        return distr


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Tuple[int, ...]', stride: 'Tuple[int, ...]', dilation: 'Tuple[int, ...]', padding: 'Tuple[int, ...]', dropout_rate: 'float'):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.chomp1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.chomp2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        if self.downsample is not None:
            inp = self.downsample(inp)
        return self.relu(x + inp)


class TCN(nn.Module):

    def __init__(self, c_in: 'int', c_out: 'int', num_hidden_dimensions: 'List[int]', hist_len: 'int', pred_len: 'int', dilation_base: 'int'=2, kernel_size: 'int'=2, embedding_dim: 'int'=10, dropout_rate: 'float'=0.2, freq: 'str'='H'):
        super(TCN, self).__init__()
        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'B': 4, 'H': 5, 'T': 6, 'S': 7}
        self.hist_len = hist_len
        self.pred_len = pred_len
        assert hist_len >= pred_len, 'history length should larger or equal to prediction length'
        self.freq = to_offset(freq)
        self.embedding_dim = embedding_dim
        layers = []
        dims = num_hidden_dimensions
        num_levels = len(dims)
        for i in range(num_levels):
            dilation_size = dilation_base ** i
            in_channels = c_in + self.embedding_dim if i == 0 else dims[i - 1]
            out_channels = dims[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout_rate=dropout_rate)]
        self.temporal_conv_net = nn.Sequential(*layers)
        self.projection = nn.Linear(dims[-1], c_out)
        self.proj_len = nn.Linear(self.hist_len, self.pred_len)
        self.time_feat_embedding = nn.Linear(freq_map[self.freq.name], self.embedding_dim)

    def create_inputs(self, x, x_mark, y_mark):
        x = torch.cat((x, torch.zeros(size=(x.shape[0], self.pred_len, x.shape[2]))), dim=1)
        mark_inp = torch.cat((x_mark, y_mark), dim=1)
        mark_inp = self.time_feat_embedding(mark_inp)
        inp = torch.cat((x, mark_inp), dim=-1)
        return inp

    def forward(self, x, x_mark, y_mark):
        x = self.create_inputs(x, x_mark, y_mark)
        x = self.temporal_conv_net(x.transpose(1, 2)).transpose(1, 2)
        x = self.projection(x)
        return x[:, -self.pred_len:, :]


class Transformer(nn.Module):

    def __init__(self, c_in: 'int', c_out: 'int', hist_len: 'int', cntx_len: 'int', pred_int: 'int', d_model: 'int'=512, n_head: 'int'=8, num_encoder_layers: 'int'=6, num_decoder_layers: 'int'=6, dim_feedforward: 'int'=2048, dropout_rate: 'float'=0.1, activation: 'str'='relu', embedding_dim: 'int'=10, freq: 'str'='H'):
        super(Transformer, self).__init__()
        freq_map = {'Y': 1, 'M': 2, 'D': 4, 'B': 4, 'H': 5, 'T': 6, 'S': 7}
        self.c_in = c_in
        self.c_out = c_out
        self.hist_len = hist_len
        self.cntx_len = cntx_len
        self.pred_len = pred_int
        self.freq = to_offset(freq)
        self.embedding_dim = embedding_dim
        self.time_feat_embedding = nn.Linear(freq_map[self.freq.name], self.embedding_dim)
        self.enc_embedding = nn.Linear(self.embedding_dim + c_in, d_model)
        self.dec_embedding = nn.Linear(self.embedding_dim + c_in, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_head, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout_rate, activation=activation)
        self.projection = nn.Linear(d_model, c_out)

    def create_inputs(self, x, x_mark, y_mark):
        mark_inp = torch.cat((x_mark, y_mark), dim=1)
        mark_inp = self.time_feat_embedding(mark_inp)
        enc_inp = torch.cat((x, mark_inp[:, :self.hist_len, :]), dim=-1)
        dec_inp = torch.cat((x[:, -self.cntx_len:, :], torch.zeros(size=(x.shape[0], self.pred_len, x.shape[2]))), dim=1)
        dec_inp = torch.cat((dec_inp, mark_inp[:, -self.cntx_len - self.pred_len:, :]), dim=-1)
        return enc_inp, dec_inp

    def forward(self, x, x_mark, y_mark):
        enc_inp, dec_inp = self.create_inputs(x, x_mark, y_mark)
        enc_inp = self.enc_embedding(enc_inp)
        dec_inp = self.dec_embedding(dec_inp)
        out = self.transformer(enc_inp.transpose(0, 1), dec_inp.transpose(0, 1))
        out = self.projection(out).transpose(0, 1)
        return out[:, -self.pred_len:, :]


class ScaledDotProductAttention(nn.Module):

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', attn_mask: 'Optional[Tensor]'=None):
        emb_dim = query.shape[-1]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = query.matmul(key.transpose(-1, -2)) / np.sqrt(emb_dim)
        if attn_mask:
            scores = scores.masked_fill(attn_mask, -np.inf)
        attn_output_weights = F.dropout(F.softmax(scores, dim=-1))
        attn_output = attn_output_weights.matmul(value)
        return attn_output, attn_output_weights


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim: 'int', num_heads: 'int', dropout_rate: 'float'=0.0, bias: 'bool'=True, kv_bias: 'bool'=False, k_dim: 'int'=None, v_dim: 'int'=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim if k_dim is not None else embed_dim
        self.v_dim = v_dim if v_dim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        self.kv_bias = kv_bias
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.q_proj_layer = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj_layer = nn.Linear(embed_dim, self.k_dim, bias=kv_bias)
        self.v_proj_layer = nn.Linear(embed_dim, self.v_dim, bias=kv_bias)
        self.attention_layer = ScaledDotProductAttention()
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.q_proj_layer.weight.data)
        xavier_uniform_(self.k_proj_layer.weight.data)
        xavier_uniform_(self.v_proj_layer.weight.data)
        if self.bias:
            constant_(self.q_proj_layer.bias.data, 0.0)
        if self.kv_bias:
            xavier_normal_(self.k_proj_layer.bias.data)
            xavier_normal_(self.v_proj_layer.bias.data)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', attn_mask: 'Optional[Tensor]'=None):
        """

        :param query: (batch_size, tgt_len, emb_dim)
        :param key: (batch_size, src_len, emb_dim)
        :param value: (batch_size, src_len, emb_dim)
        :param attn_mask: (tgt_len, src_len)

        :return:
        :output attn_output: (batch_size, num_heads, tgt_len, emb_dim)
        :output attn_output_weights: (batch_size, num_heads, tgt_len, src_len)
        """
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        num_heads = self.num_heads
        q = self.q_proj_layer(query).view(batch_size, tgt_len, num_heads, -1)
        k = self.k_proj_layer(key).view(batch_size, src_len, num_heads, -1)
        v = self.v_proj_layer(value).view(batch_size, src_len, num_heads, -1)
        return self.attention_layer(q, k, v, attn_mask)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaLayer,
     lambda: ([], {'function': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TokenLinearEmbedding,
     lambda: ([], {'c_in': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_BTDLOZC_SJTU_TimeSeriesResearch(_paritybench_base):
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

