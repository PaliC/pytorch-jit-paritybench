import sys
_module = sys.modules[__name__]
del sys
convlstm = _module
convlstm_decoder = _module

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


import torch.nn as nn


import torch


import math


class HadamardProduct(nn.Module):

    def __init__(self, shape):
        super(HadamardProduct, self).__init__()
        self.weights = nn.Parameter(torch.rand(shape))

    def forward(self, x):
        return x * self.weights


class ConvLSTMCell(nn.Module):

    def __init__(self, img_size, input_dim, hidden_dim, kernel_size, cnn_dropout, rnn_dropout, bias=True, peephole=False, layer_norm=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel for both cnn and rnn.
        cnn_dropout, rnn_dropout: float
            cnn_dropout: dropout rate for convolutional input.
            rnn_dropout: dropout rate for convolutional state.
        bias: bool
            Whether or not to add the bias.
        peephole: bool
            add connection between cell state to gates
        layer_norm: bool
            layer normalization 
        """
        super(ConvLSTMCell, self).__init__()
        self.input_shape = img_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size[0] / 2), int(self.kernel_size[1] / 2)
        self.stride = 1, 1
        self.bias = bias
        self.peephole = peephole
        self.layer_norm = layer_norm
        self.out_height = int((self.input_shape[0] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
        self.out_width = int((self.input_shape[1] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1)
        self.input_conv = nn.Conv2d(in_channels=self.input_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        self.rnn_conv = nn.Conv2d(self.hidden_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, padding=(math.floor(self.kernel_size[0] / 2), math.floor(self.kernel_size[1] / 2)), bias=self.bias)
        if self.peephole is True:
            self.weight_ci = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.weight_cf = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.weight_co = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.layer_norm_ci = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
            self.layer_norm_cf = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
            self.layer_norm_co = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.layer_norm_x = nn.LayerNorm([4 * self.hidden_dim, self.out_height, self.out_width])
        self.layer_norm_h = nn.LayerNorm([4 * self.hidden_dim, self.out_height, self.out_width])
        self.layer_norm_cnext = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        x = self.cnn_dropout(input_tensor)
        x_conv = self.input_conv(x)
        if self.layer_norm is True:
            x_conv = self.layer_norm_x(x_conv)
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_dim, dim=1)
        h = self.rnn_dropout(h_cur)
        h_conv = self.rnn_conv(h)
        if self.layer_norm is True:
            h_conv = self.layer_norm_h(h_conv)
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_dim, dim=1)
        if self.peephole is True:
            f = torch.sigmoid(x_f + h_f + self.layer_norm_cf(self.weight_cf(c_cur)) if self.layer_norm is True else self.weight_cf(c_cur))
            i = torch.sigmoid(x_i + h_i + self.layer_norm_ci(self.weight_ci(c_cur)) if self.layer_norm is True else self.weight_ci(c_cur))
        else:
            f = torch.sigmoid(x_f + h_f)
            i = torch.sigmoid(x_i + h_i)
        g = torch.tanh(x_c + h_c)
        c_next = f * c_cur + i * g
        if self.peephole is True:
            o = torch.sigmoid(x_o + h_o + self.layer_norm_co(self.weight_co(c_cur)) if self.layer_norm is True else self.weight_co(c_cur))
        else:
            o = torch.sigmoid(x_o + h_o)
        if self.layer_norm is True:
            c_next = self.layer_norm_cnext(c_next)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size):
        height, width = self.out_height, self.out_width
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device), torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device)


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        cnn_dropout, rnn_dropout: float
            cnn_dropout: dropout rate for convolutional input.
            rnn_dropout: dropout rate for convolutional state.
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_sequence: return output sequence or final output only
        bidirectional: bool
            bidirectional ConvLSTM
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two sequences output and state
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(input_dim=64, hidden_dim=16, kernel_size=(3, 3), 
                               cnn_dropout = 0.2,
                               rnn_dropout=0.2, batch_first=True, bias=False)
        >> output, last_state = convlstm(x)
    """

    def __init__(self, img_size, input_dim, hidden_dim, kernel_size, cnn_dropout=0.5, rnn_dropout=0.5, batch_first=False, bias=True, peephole=False, layer_norm=False, return_sequence=True, bidirectional=False):
        super(ConvLSTM, self).__init__()
        None
        self.batch_first = batch_first
        self.return_sequence = return_sequence
        self.bidirectional = bidirectional
        cell_fw = ConvLSTMCell(img_size=img_size, input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, cnn_dropout=cnn_dropout, rnn_dropout=rnn_dropout, bias=bias, peephole=peephole, layer_norm=layer_norm)
        self.cell_fw = cell_fw
        if self.bidirectional is True:
            cell_bw = ConvLSTMCell(img_size=img_size, input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, cnn_dropout=cnn_dropout, rnn_dropout=rnn_dropout, bias=bias, peephole=peephole, layer_norm=layer_norm)
            self.cell_bw = cell_bw

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        layer_output, last_state
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, seq_len, _, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state, hidden_state_inv = self._init_hidden(batch_size=b)
        input_fw = input_tensor
        h, c = hidden_state
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell_fw(input_tensor=input_fw[:, t, :, :, :], cur_state=[h, c])
            output_inner.append(h)
        output_inner = torch.stack(output_inner, dim=1)
        layer_output = output_inner
        last_state = [h, c]
        if self.bidirectional is True:
            input_inv = input_tensor
            h_inv, c_inv = hidden_state_inv
            output_inv = []
            for t in range(seq_len - 1, -1, -1):
                h_inv, c_inv = self.cell_bw(input_tensor=input_inv[:, t, :, :, :], cur_state=[h_inv, c_inv])
                output_inv.append(h_inv)
            output_inv.reverse()
            output_inv = torch.stack(output_inv, dim=1)
            layer_output = torch.cat((output_inner, output_inv), dim=2)
            last_state_inv = [h_inv, c_inv]
        return layer_output if self.return_sequence is True else layer_output[:, -1:], last_state, last_state_inv if self.bidirectional is True else None

    def _init_hidden(self, batch_size):
        init_states_fw = self.cell_fw.init_hidden(batch_size)
        init_states_bw = None
        if self.bidirectional is True:
            init_states_bw = self.cell_bw.init_hidden(batch_size)
        return init_states_fw, init_states_bw


class Flatten(torch.nn.Module):

    def forward(self, input):
        b, seq_len, _, h, w = input.size()
        return input.view(b, seq_len, -1)


class ConvLSTMNetwork(torch.nn.Module):

    def __init__(self, img_size_list, input_channel, hidden_channels, kernel_size, num_layers, bidirectional=False):
        super(ConvLSTMNetwork, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        convlstm_layer = []
        for i in range(num_layers):
            layer = convlstm.ConvLSTM(img_size_list[i], input_channel, hidden_channels[i], kernel_size[i], 0.2, 0.0, batch_first=True, bias=True, peephole=True, layer_norm=True, return_sequence=config.SEQUENCE_OUTPUT, bidirectional=self.bidirectional)
            convlstm_layer.append(layer)
            input_channel = hidden_channels[i] * (2 if self.bidirectional else 1)
        self.convlstm_layer = torch.nn.ModuleList(convlstm_layer)
        self.flatten = Flatten()
        self.linear2 = torch.nn.Linear(hidden_channels[-1] * (2 if self.bidirectional else 1) * 16, 2)

    def forward(self, x):
        input_tensor = x
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
        out_flatten = self.flatten(input_tensor)
        output = self.linear2(out_flatten)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (HadamardProduct,
     lambda: ([], {'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_KimUyen_ConvLSTM_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

