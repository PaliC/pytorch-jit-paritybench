import sys
_module = sys.modules[__name__]
del sys
infer = _module
make_rttm = _module
model_averaging = _module
train = _module
yaml2bash = _module
feature = _module
kaldi_data = _module
diarization_dataset = _module
infer = _module
loss = _module
models = _module
train = _module

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


from collections import OrderedDict


import numpy as np


from scipy.ndimage import shift


import torch.nn as nn


import torch.nn.functional as F


from itertools import permutations


import math


from torch.nn import TransformerEncoder


from torch.nn import TransformerEncoderLayer


from torch.optim.lr_scheduler import _LRScheduler


from torch.optim import Optimizer


import logging


from torch import optim


from torch import nn


from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        	ext{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        	ext{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        	ext{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=False):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerModel, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos
        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(n_units, n_speakers)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=False, activation=None):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1))
                self.src_mask = mask
        else:
            self.src_mask = None
        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        src = src.transpose(0, 1)
        if self.has_pos:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.transpose(0, 1)
        output = self.decoder(output)
        if activation:
            output = activation(output)
        output = [out[:ilen] for out, ilen in zip(output, ilens)]
        return output

    def get_attention_weight(self, src):
        attn_weight = []

        def hook(module, input, output):
            attn_weight.append(output[1])
        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))
        self.eval()
        with torch.no_grad():
            self.forward(src)
        for handle in handles:
            handle.remove()
        self.train()
        return torch.stack(attn_weight)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerModel,
     lambda: ([], {'n_speakers': 4, 'in_size': 4, 'n_heads': 4, 'n_units': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_Xflick_EEND_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

