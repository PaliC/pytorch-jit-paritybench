import sys
_module = sys.modules[__name__]
del sys
_init_ = _module
complexNN = _module
functional = _module
nn = _module
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


import numpy as np


import torch


from torch.nn.functional import dropout


from torch.nn.functional import dropout2d


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


from torch.nn.functional import avg_pool1d


from torch.nn.functional import max_pool1d


from torch.nn.functional import gelu


from torch.nn.functional import leaky_relu


from torch.nn.functional import elu


from torch.nn.functional import softmax


from torch.nn import CosineSimilarity


from torch import relu


from torch import tanh


from torch import sigmoid


import torch.nn as nn


def complexRelu(inp):
    return torch.complex(relu(inp.real), relu(inp.imag))


class cRelu(nn.Module):

    @staticmethod
    def forward(inp):
        return complexRelu(inp)


def complexElu(inp):
    return torch.complex(elu(inp.real), elu(inp.imag))


class cElu(nn.Module):

    @staticmethod
    def forward(inp):
        return complexElu(inp)


def complexLeakyRelu(inp):
    return torch.complex(leaky_relu(inp.real), leaky_relu(inp.imag))


class cLeakyRelu(nn.Module):

    @staticmethod
    def forward(inp):
        return complexLeakyRelu(inp)


def complexSoftmax(inp):
    return torch.complex(softmax(inp.real), softmax(inp.imag))


class cSoftmax(nn.Module):

    @staticmethod
    def forward(inp):
        return complexSoftmax(inp)


def complexGelu(inp):
    return torch.complex(gelu(inp.real), gelu(inp.imag))


class cGelu(nn.Module):

    @staticmethod
    def forward(inp):
        return complexGelu(inp)


def complexTanh(inp):
    return torch.complex(tanh(inp.real), tanh(inp.imag))


class cTanh(nn.Module):

    @staticmethod
    def forward(inp):
        return complexTanh(inp)


def complexSigmoid(inp):
    return torch.complex(sigmoid(inp.real), sigmoid(inp.imag))


class cSigmoid(nn.Module):

    @staticmethod
    def forward(inp):
        return complexSigmoid(inp)


class cBatchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.real_bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.imag_bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, inp):
        real_input = torch.real(inp)
        imag_input = torch.imag(inp)
        real_output = self.real_bn(real_input)
        imag_output = self.imag_bn(imag_input)
        return torch.complex(real_output, imag_output)


class cBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.real_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.imag_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, inp):
        real_input = torch.real(inp)
        imag_input = torch.imag(inp)
        real_output = self.real_bn(real_input)
        imag_output = self.imag_bn(imag_input)
        return torch.complex(real_output, imag_output)


class cBatchNorm3d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.real_bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.imag_bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, inp):
        real_input = torch.real(inp)
        imag_input = torch.imag(inp)
        real_output = self.real_bn(real_input)
        imag_output = self.imag_bn(imag_input)
        return torch.complex(real_output, imag_output)


class cLayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, elementwise_affine=False):
        super().__init__()
        self.real_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
        self.imag_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inp):
        real_input = torch.real(inp)
        imag_input = torch.imag(inp)
        real_output = self.real_norm(real_input)
        imag_output = self.imag_norm(imag_input)
        return torch.complex(real_output, imag_output)


def complexDropout(inp, p=0.5, training=True):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


class cDropout(nn.Module):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complexDropout(inp, self.p)
        else:
            return inp


def complexDropout2d(inp, p=0.5, training=True):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout2d(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


class cDropout2d(nn.Module):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complexDropout2d(inp, self.p)
        else:
            return inp


def _retrieve_elements_from_indices(tensor, indices):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output


def complexMaxPool2d(inp, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    absolute_value, indices = max_pool2d(inp.abs(), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=True)
    absolute_value = absolute_value.type(torch.complex64)
    angle = torch.atan2(inp.imag, inp.real)
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value * (torch.cos(angle).type(torch.complex64) + 1.0j * torch.sin(angle).type(torch.complex64))


class cMaxPool2d(nn.Module):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complexMaxPool2d(inp, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices)


def complexAvgPool2d(inp, *args, **kwargs):
    """
        copy from https://github.com/wavefrontshaping/complexPyTorch
        """
    absolute_value_real = avg_pool2d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(inp.imag, *args, **kwargs)
    return absolute_value_real.type(torch.complex64) + 1.0j * absolute_value_imag.type(torch.complex64)


class cAvgPool2d(torch.nn.Module):
    """
    copy from https://github.com/wavefrontshaping/complexPyTorch
        """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complexAvgPool2d(inp, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad, divisor_override=self.divisor_override)


class cMaxPool1d(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complexMaxPool2d(inp, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, ceil_mode=self.ceil_mode, return_indices=self.return_indices)


def complexAvgPool1d(inp, *args, **kwargs):
    absolute_value_real = avg_pool1d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool1d(inp.imag, *args, **kwargs)
    return absolute_value_real.type(torch.complex64) + 1.0j * absolute_value_imag.type(torch.complex64)


class cAvgPool1d(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complexAvgPool1d(inp, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad, divisor_override=self.divisor_override)


class cLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias)
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.xavier_uniform_(self.bias)

    def forward(self, inp):
        if not inp.dtype == torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        return torch.matmul(inp, self.weight.T) + self.bias


class cMLP(nn.Module):
    """
    Complex Multilayer Perceptron
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.input_layer = cLinear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([cLinear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = cLinear(hidden_size, output_size)
        self.activation = cGelu()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for i in range(self.num_layers - 1):
            x = self.activation(self.hidden_layers[i](x))
        output = self.output_layer(x)
        return output


class cConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=1, groups=1):
        super().__init__()
        assert in_channels % groups == 0, 'In_channels should be an integer multiple of groups.'
        assert out_channels % groups == 0, 'Out_channels should be an integer multiple of groups.'
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat)) if bias else None
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels // groups, kernel_size), dtype=torch.cfloat))

    def forward(self, inp):
        if not inp.dtype == torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        out = torch.nn.functional.conv1d(input=inp, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return out


class cConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=1, groups=1):
        super().__init__()
        assert in_channels % groups == 0, 'In_channels should be an integer multiple of groups.'
        assert out_channels % groups == 0, 'Out_channels should be an integer multiple of groups.'
        if isinstance(padding, int):
            padding = padding, padding
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels // groups, *kernel_size), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat)) if bias else None

    def forward(self, inp):
        if not inp.dtype == torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        out = torch.nn.functional.conv2d(input=inp, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return out


class cRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.Wx = cLinear(input_size, hidden_size, bias)
        self.Wh = cLinear(hidden_size, hidden_size, bias)

    def forward(self, x, h_prev):
        h = complexTanh(self.Wx(x) + self.Wh(h_prev))
        return h


class cGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.W_r = cLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_z = cLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_h = cLinear(input_size + hidden_size, hidden_size, bias=bias)

    def forward(self, x, h_prev):
        r = complexSigmoid(self.W_r(torch.cat((x, h_prev), 1)))
        z = complexSigmoid(self.W_z(torch.cat((x, h_prev), 1)))
        h_hat = complexTanh(self.W_h(torch.cat((x, r * h_prev), 1)))
        h_new = (1 - z) * h_hat + z * h_prev
        return h_new


class cLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.W_i = cLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_f = cLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_c = cLinear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_o = cLinear(input_size + hidden_size, hidden_size, bias=bias)

    def forward(self, x, hidden):
        h, c = hidden
        i = complexSigmoid(self.W_i(torch.cat((x, h), 1)))
        f = complexSigmoid(self.W_f(torch.cat((x, h), 1)))
        g = complexTanh(self.W_c(torch.cat((x, h), 1)))
        o = complexSigmoid(self.W_o(torch.cat((x, h), 1)))
        c = f * c + i * g
        h = o * complexTanh(c)
        return h, c


class cRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        """
        :param input_size: (int) Dimension of inputs
        :param hidden_size: (int) Dimension of hidden states
        :param num_layers: (int)  Number of recurrent layers
        """
        super().__init__()
        rnn_cells = [cRNNCell(input_size, hidden_size, bias)]
        rnn_cells += [cRNNCell(hidden_size, hidden_size, bias) for _ in range(num_layers - 1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)

    def forward(self, sequence, init_states=None):
        """
        :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
        :param init_states: (torch.Tensor, shape=[num_layers, batch, hidden_size]) :math:`h_0`
        :returns:
            - output: (seq_len, batch, hidden_size) Sequence of outputs
            - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.
        """
        assert len(sequence.shape) == 3, f'RNN takes order 3 tensor with shape=(seq_len, nsamples, {self.insize})'
        final_hidden = []
        for h, cell in zip(init_states, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h = cell(cell_input, h)
                states.append(h)
            sequence = torch.stack(states)
            final_hidden.append(h)
        assert torch.equal(sequence[-1, :, :], final_hidden[-1])
        return sequence, torch.stack(final_hidden)


class cGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        """
        :param input_size: (int) Dimension of inputs
        :param hidden_size: (int) Dimension of hidden states
        :param num_layers: (int)  Number of recurrent layers
        """
        super().__init__()
        rnn_cells = [cGRUCell(input_size, hidden_size, bias)]
        rnn_cells += [cGRUCell(hidden_size, hidden_size, bias) for _ in range(num_layers - 1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)

    def forward(self, sequence, init_states=None):
        """
        :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
        :param init_states: (torch.Tensor, shape=[num_layers, batch, hidden_size]) :math:`h_0`
        :returns:
            - output: (seq_len, batch, hidden_size) Sequence of outputs
            - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.
        """
        assert len(sequence.shape) == 3, f'GRU takes order 3 tensor with shape=(seq_len, nsamples, {self.insize})'
        final_hidden = []
        for h, cell in zip(init_states, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h = cell(cell_input, h)
                states.append(h)
            sequence = torch.stack(states)
            final_hidden.append(h)
        assert torch.equal(sequence[-1, :, :], final_hidden[-1])
        return sequence, torch.stack(final_hidden)


class cLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        """
        :param input_size: (int) Dimension of inputs
        :param hidden_size: (int) Dimension of hidden states
        :param num_layers: (int)  Number of recurrent layers
        """
        super().__init__()
        rnn_cells = [cLSTMCell(input_size, hidden_size, bias)]
        rnn_cells += [cLSTMCell(hidden_size, hidden_size, bias) for _ in range(num_layers - 1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)

    def forward(self, sequence, init_states):
        """
        :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
        :param init_states: (tuple, shape=([num_layers, batch, hidden_size], [num_layers, batch, hidden_size])) :math:`(h_0, c_0)`
        :returns:
            - output: (seq_len, batch, hidden_size) Sequence of outputs
            - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.
        """
        assert len(sequence.shape) == 3, f'LSTM takes order 3 tensor with shape=(seq_len, nsamples, {self.insize})'
        final_hidden = []
        h0, c0 = init_states
        for h, c, cell in zip(h0, c0, self.rnn_cells):
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                h, c = cell(cell_input, (h, c))
                states.append(h)
            sequence = torch.stack(states)
            final_hidden.append(h)
        assert torch.equal(sequence[-1, :, :], final_hidden[-1])
        return sequence, torch.stack(final_hidden)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (cConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (cConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (cDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (cDropout2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (cGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (cLSTMCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), (torch.rand([4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (cLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (cMLP,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (cRNNCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_XinyuanLiao_ComplexNN(_paritybench_base):
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

