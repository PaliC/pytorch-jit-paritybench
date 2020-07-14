import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
data = _module
an4 = _module
common_voice = _module
data_loader = _module
librispeech = _module
merge_manifests = _module
sparse_image_warp = _module
spec_augment = _module
ted = _module
utils = _module
voxforge = _module
decoder = _module
logger = _module
model = _module
multiproc = _module
noise_inject = _module
opts = _module
search_lm_params = _module
select_lm_params = _module
server = _module
test = _module
train = _module
transcribe = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.utils.data.distributed


from torch.distributed import get_rank


from torch.distributed import get_world_size


from torch.utils.data.sampler import Sampler


import numpy as np


import scipy.signal


from scipy.io.wavfile import read


import math


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import random


import matplotlib


import matplotlib.pyplot as plt


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from scipy.io.wavfile import write


import logging


import warnings


class SequenceWise(nn.Module):

    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):

    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask
            for i, length in enumerate(lengths):
                length = length.item()
                if mask[i].size(2) - length > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):

    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class Lookahead(nn.Module):

    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = 0, self.context - 1
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1, groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'n_features=' + str(self.n_features) + ', context=' + str(self.context) + ')'


supported_rnns = {'lstm': nn.LSTM, 'rnn': nn.RNN, 'gru': nn.GRU}


supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class DeepSpeech(nn.Module):

    def __init__(self, rnn_type=nn.LSTM, labels='abc', rnn_hidden_size=768, nb_layers=5, audio_conf=None, bidirectional=True, context=20):
        super(DeepSpeech, self).__init__()
        if audio_conf is None:
            audio_conf = {}
        self.version = '0.0.1'
        self.hidden_size = rnn_hidden_size
        self.hidden_layers = nb_layers
        self.rnn_type = rnn_type
        self.audio_conf = audio_conf or {}
        self.labels = labels
        self.bidirectional = bidirectional
        sample_rate = self.audio_conf.get('sample_rate', 16000)
        window_size = self.audio_conf.get('window_size', 0.02)
        num_classes = len(self.labels)
        self.conv = MaskConv(nn.Sequential(nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)), nn.BatchNorm2d(32), nn.Hardtanh(0, 20, inplace=True), nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)), nn.BatchNorm2d(32), nn.Hardtanh(0, 20, inplace=True)))
        rnn_input_size = int(math.floor(sample_rate * window_size / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(Lookahead(rnn_hidden_size, context=context), nn.Hardtanh(0, 20, inplace=True)) if not bidirectional else None
        fully_connected = nn.Sequential(nn.BatchNorm1d(rnn_hidden_size), nn.Linear(rnn_hidden_size, num_classes, bias=False))
        self.fc = nn.Sequential(SequenceWise(fully_connected))
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        for rnn in self.rnns:
            x = rnn(x, output_lengths)
        if not self.bidirectional:
            x = self.lookahead(x)
        x = self.fc(x)
        x = x.transpose(0, 1)
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = (seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1
        return seq_len.int()

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'], labels=package['labels'], audio_conf=package['audio_conf'], rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'], labels=package['labels'], audio_conf=package['audio_conf'], rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer=None, amp=None, epoch=None, iteration=None, loss_results=None, cer_results=None, wer_results=None, avg_loss=None, meta=None):
        package = {'version': model.version, 'hidden_size': model.hidden_size, 'hidden_layers': model.hidden_layers, 'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()), 'audio_conf': model.audio_conf, 'labels': model.labels, 'state_dict': model.state_dict(), 'bidirectional': model.bidirectional}
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if amp is not None:
            package['amp'] = amp.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (InferenceBatchSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lookahead,
     lambda: ([], {'n_features': 4, 'context': 4}),
     lambda: ([torch.rand([4, 1, 4])], {}),
     False),
    (SequenceWise,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_SeanNaren_deepspeech_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

