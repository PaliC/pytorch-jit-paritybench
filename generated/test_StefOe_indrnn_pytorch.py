import sys
_module = sys.modules[__name__]
del sys
addition_test = _module
indrnn = _module
seq_mnist_test = _module
test = _module

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


import numpy as np


from time import time


from torch.nn import Parameter


from torch.nn import ParameterList


import math


from torchvision import datasets


from torchvision import transforms


from torch.autograd import Variable


class IndRNNCell(nn.Module):
    """An IndRNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \\tanh(w_{ih} * x + b_{ih}  +  w_{hh} (*) h)
    With (*) being element-wise vector multiplication.
    If nonlinearity='relu', then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'relu'
        hidden_min_abs: Minimal absolute inital value for hidden weights. Default: 0
        hidden_max_abs: Maximal absolute inital value for hidden weights. Default: None

    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(input_size x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`

    Examples::

        >>> rnn = nn.IndRNNCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """
    __constants__ = ['hidden_max_abs', 'hidden_min_abs', 'input_size', 'hidden_size', 'nonlinearity', 'hidden_init', 'recurrent_init']

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='relu', hidden_min_abs=0, hidden_max_abs=None, hidden_init=None, recurrent_init=None, gradient_clip=None):
        super(IndRNNCell, self).__init__()
        self.hidden_max_abs = hidden_max_abs
        self.hidden_min_abs = hidden_min_abs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.hidden_init = hidden_init
        self.recurrent_init = recurrent_init
        if self.nonlinearity == 'tanh':
            self.activation = F.tanh
        elif self.nonlinearity == 'relu':
            self.activation = F.relu
        else:
            raise RuntimeError('Unknown nonlinearity: {}'.format(self.nonlinearity))
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
        if gradient_clip:
            if isinstance(gradient_clip, tuple):
                min_g, max_g = gradient_clip
            else:
                max_g = gradient_clip
                min_g = -max_g
            self.weight_ih.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            self.weight_hh.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            if bias:
                self.bias_ih.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
        self.reset_parameters()

    def check_bounds(self):
        self.weight_hh.data = check_bounds(self.weight_hh.data, self.hidden_min_abs, self.hidden_max_abs)

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if 'bias' in name:
                weight.data.zero_()
            elif 'weight_hh' in name:
                if self.recurrent_init is None:
                    nn.init.constant_(weight, 1)
                else:
                    self.recurrent_init(weight)
            elif 'weight_ih' in name:
                if self.hidden_init is None:
                    nn.init.normal_(weight, 0, 0.01)
                else:
                    self.hidden_init(weight)
            else:
                weight.data.normal_(0, 0.01)
        self.check_bounds()

    def forward(self, input, hx):
        return self.activation(F.linear(input, self.weight_ih, self.bias_ih) + torch.mul(self.weight_hh, hx))


def check_bounds(weight, min_abs, max_abs):
    if min_abs:
        abs_kernel = torch.abs(weight).clamp_(min=min_abs)
        weight = torch.mul(torch.sign(weight), abs_kernel)
    if max_abs:
        weight = weight.clamp(max=max_abs, min=-max_abs)
    return weight


class IndRNN(nn.Module):
    """Applies a multi-layer IndRNN with `tanh` or `ReLU` non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

        h_t = \\tanh(w_{ih} x_t + b_{ih}  +  w_{hh} (*) h_{(t-1)})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`input_t`
    for the first layer. (*) is element-wise multiplication.
    If :attr:`nonlinearity`='relu', then `ReLU` is used instead of `tanh`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        hidden_inits: The init value generator for the hidden unit.
        recurrent_inits: The init value generator for the recurrent unit.
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_norm: If ``True``, then batch normalization is applied after each time step
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)`

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size * num_directions)`: tensor
          containing the output features (`h_k`) from the last layer of the RNN,
          for each `k`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for `k = seq_len`.

    Attributes:
        cells[k]: individual IndRNNCells containing the weights

    Examples::

        >>> rnn = nn.IndRNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output = rnn(input, h0)
    """

    def __init__(self, input_size, hidden_size, n_layer=1, batch_norm=False, batch_first=False, bidirectional=False, hidden_inits=None, recurrent_inits=None, **kwargs):
        super(IndRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.n_layer = n_layer
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = num_directions = 2 if self.bidirectional else 1
        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1
        cells = []
        cells_bi = []
        for i in range(n_layer):
            if recurrent_inits is not None:
                kwargs['recurrent_init'] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs['hidden_init'] = hidden_inits[i]
            in_size = input_size if i == 0 else hidden_size * num_directions
            cells.append(IndRNNCell(in_size, hidden_size, **kwargs))
            cells_bi.append(IndRNNCell(in_size, hidden_size, **kwargs))
        self.cells = nn.ModuleList(cells)
        self.cells_bi = nn.ModuleList(cells_bi)
        if batch_norm:
            bns = []
            for i in range(n_layer):
                bns.append(nn.BatchNorm1d(hidden_size * num_directions))
            self.bns = nn.ModuleList(bns)
        h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        self.register_buffer('h0', h0)

    def forward(self, x, hidden=torch.tensor(float('nan'))):
        batch_norm = self.batch_norm
        time_index = self.time_index
        batch_index = self.batch_index
        num_directions = self.num_directions
        hiddens = []
        i = 0
        for cell in self.cells:
            hx = self.h0.unsqueeze(0).expand(x.size(batch_index), self.hidden_size * num_directions).contiguous()
            x_n = []
            hx_cell = hx[:, :self.hidden_size * 1]
            hx_cell_bi = hx[:, self.hidden_size:self.hidden_size * 2]
            cell.weight_hh.data = check_bounds(cell.weight_hh.data, cell.hidden_min_abs, cell.hidden_max_abs)
            outputs = []
            x_T = torch.unbind(x, time_index)
            time_frame = len(x_T)
            for t in range(time_frame):
                hx_cell = cell(x_T[t], hx_cell)
                outputs.append(hx_cell)
            x_cell = torch.stack(outputs, time_index)
            if self.bidirectional:
                outputs_bi = []
                for t in range(time_frame - 1, -1, -1):
                    hx_cell_bi = self.cells_bi[i](x_T[t], hx_cell_bi)
                    outputs_bi.append(hx_cell_bi)
                x_cell_bi = torch.stack(outputs_bi[::-1], time_index)
                x_cell = torch.cat([x_cell, x_cell_bi], 2)
            x_n.append(x_cell)
            hiddens.append(hx_cell)
            x = torch.cat(x_n, -1)
            if batch_norm:
                if self.batch_first:
                    x = self.bns[i](x.permute(batch_index, 2, time_index).contiguous()).permute(0, 2, 1)
                else:
                    x = self.bns[i](x.permute(batch_index, 2, time_index).contiguous()).permute(2, 0, 1)
            i += 1
        return x.squeeze(2), torch.cat(hiddens, -1)


TIME_STEPS = 784


RECURRENT_MAX = pow(2, 1 / TIME_STEPS)


RECURRENT_MIN = pow(1 / 2, 1 / TIME_STEPS)


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, n_layer=2, model=IndRNN):
        super(Net, self).__init__()
        recurrent_inits = []
        for _ in range(n_layer - 1):
            recurrent_inits.append(lambda w: nn.init.uniform_(w, 0, RECURRENT_MAX))
        recurrent_inits.append(lambda w: nn.init.uniform_(w, RECURRENT_MIN, RECURRENT_MAX))
        self.indrnn = model(input_size, hidden_size, n_layer, batch_norm=args.batch_norm, hidden_max_abs=RECURRENT_MAX, batch_first=True, bidirectional=args.bidirectional, recurrent_inits=recurrent_inits, gradient_clip=5)
        self.lin = nn.Linear(hidden_size * 2 if args.bidirectional else hidden_size, 10)
        self.lin.bias.data.fill_(0.1)
        self.lin.weight.data.normal_(0, 0.01)

    def forward(self, x, hidden=None):
        y, _ = self.indrnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.cell1 = nn.LSTM(2, args.hidden_size)
        self.lin = nn.Linear(args.hidden_size, 1)

    def forward(self, x, hidden=None):
        x, hidden = self.cell1(x, hidden)
        return self.lin(x[-1]).squeeze(1)


class IndRNNv2(nn.Module):
    """Applies a multi-layer IndRNN with `tanh` or `ReLU` non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

        h_t = \\tanh(w_{ih} x_t + b_{ih}  +  w_{hh} (*) h_{(t-1)})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`input_t`
    for the first layer. (*) is element-wise multiplication.
    If :attr:`nonlinearity`='relu', then `ReLU` is used instead of `tanh`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        hidden_inits: The init value generator for the hidden unit.
        recurrent_inits: The init value generator for the recurrent unit.
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_norm: If ``True``, then batch normalization is applied after each time step
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)`
        hidden_min_abs: Minimal absolute inital value for hidden weights. Default: 0
        hidden_max_abs: Maximal absolute inital value for hidden weights. Default: None

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size * num_directions)`: tensor
          containing the output features (`h_k`) from the last layer of the RNN,
          for each `k`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for `k = seq_len`.

    Attributes:
        cells[k]: individual IndRNNCells containing the weights

    Examples::

        >>> rnn = nn.IndRNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output = rnn(input, h0)
    """

    def __init__(self, input_size, hidden_size, n_layer=1, batch_norm=False, batch_first=False, bidirectional=False, bias=True, hidden_inits=None, recurrent_inits=None, nonlinearity='relu', hidden_min_abs=0, hidden_max_abs=None, gradient_clip=None):
        super(IndRNNv2, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.n_layer = n_layer
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.hidden_min_abs = hidden_min_abs
        self.hidden_max_abs = hidden_max_abs
        self.gradient_clip = gradient_clip
        if gradient_clip:
            if isinstance(gradient_clip, tuple):
                min_g, max_g = gradient_clip
            else:
                max_g = gradient_clip
                min_g = -max_g
        if self.nonlinearity == 'tanh':
            self.activation = F.tanh
        elif self.nonlinearity == 'relu':
            self.activation = F.relu
        else:
            raise RuntimeError('Unknown nonlinearity: {}'.format(self.nonlinearity))
        self.num_directions = num_directions = 2 if self.bidirectional else 1
        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1
        self.cells_recurrent = ParameterList([Parameter(torch.Tensor(num_directions * hidden_size)) for i in range(n_layer)])
        if gradient_clip:
            for param in self.cells_recurrent:
                param.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
        cells_hidden = []
        for i in range(n_layer):
            directions_hidden = []
            in_size = input_size * num_directions if i == 0 else hidden_size * num_directions ** 2
            hidden = nn.Conv1d(in_size, hidden_size * num_directions, 1, groups=num_directions)
            if hidden_inits is not None:
                hidden_inits[i](hidden.weight)
            else:
                torch.nn.init.normal_(hidden.weight, 0, 0.01)
            if bias:
                torch.nn.init.constant_(hidden.bias, 0)
                if gradient_clip:
                    hidden.bias.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            if recurrent_inits is not None:
                recurrent_inits[i](self.cells_recurrent[i])
            else:
                torch.nn.init.constant_(self.cells_recurrent[i], 1)
            hidden.weight.data = check_bounds(hidden.weight.data, self.hidden_min_abs, self.hidden_max_abs)
            if gradient_clip:
                hidden.weight.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            cells_hidden.append(hidden)
        self.cells_hidden = nn.ModuleList(cells_hidden)
        if batch_norm:
            bns = []
            for i in range(n_layer):
                bns.append(nn.BatchNorm1d(hidden_size * num_directions))
            self.bns = nn.ModuleList(bns)
        h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        self.register_buffer('h0', h0)

    def forward(self, x, hidden=None):
        batch_norm = self.batch_norm
        time_index = self.time_index
        batch_index = self.batch_index
        num_directions = self.num_directions
        frame_size = x.size(self.time_index)
        batch_size = x.size(self.batch_index)
        x = x.permute(self.batch_index, -1, self.time_index)
        hiddens = []
        i = 0
        for cell_hidden in self.cells_hidden:
            cell_hidden.weight.data = check_bounds(cell_hidden.weight.data, self.hidden_min_abs, self.hidden_max_abs)
            if hidden is None:
                hx = self.h0.unsqueeze(0).expand(batch_size, self.hidden_size * num_directions).contiguous()
            else:
                hx = hidden[i]
            outputs = []
            if self.bidirectional:
                x_T = torch.cat([x, x.flip(-1)], 1)
            else:
                x_T = x
            lin = cell_hidden(x_T)
            lin = torch.unbind(lin, 2)
            recurrent_h = self.cells_recurrent[i]
            for t in range(frame_size):
                hx = self.activation(lin[t] + torch.mul(recurrent_h, hx))
                outputs.append(hx)
            x = torch.stack(outputs, 2)
            hiddens.append(hx)
            if batch_norm:
                if self.batch_first:
                    x = self.bns[i](x)
                else:
                    x = self.bns[i](x)
            i += 1
        hiddens = torch.cat(hiddens, -1)
        if self.batch_first:
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(2, 0, 1)
        return x.squeeze(2), hiddens


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (IndRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IndRNNCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IndRNNv2,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_StefOe_indrnn_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

