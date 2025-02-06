import sys
_module = sys.modules[__name__]
del sys
diarization_dataset = _module
diarization_dataset_predict = _module
feature = _module
kaldi_data = _module
dia_pred = _module
metrics = _module
offl_tfm_enc_lstm_enc_dec = _module
onl_tfm_enc_1dcnn_enc_linear_non_autoreg_pos_enc_l2norm = _module
merge_tfm_encoder = _module
streaming_tfm = _module
offl_tfm_lstm = _module
oln_tfm_enc_dec = _module
oln_tfm_enc_dec_spk_pit = _module
tfm_STB = _module
loss = _module
make_rttm = _module
utils = _module
train_STB = _module
train_dia = _module
train_dia_fintn_ch = _module
train_offl_eend_eda = _module
avg_ckpt = _module
scheduler = _module
gen_h5_output = _module

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


import numpy as np


import scipy.signal


import warnings


import torch.nn.functional as F


from scipy.signal import medfilt


import torch.nn as nn


import math


from torch.nn import TransformerEncoder


from torch.nn import TransformerEncoderLayer


from torch import Tensor


from torch.nn import TransformerDecoder


from torch.nn import TransformerDecoderLayer


from typing import Optional


from typing import Any


from typing import Union


from typing import Callable


import copy


from torch.nn.modules import Module


from torch.nn.modules.activation import MultiheadAttention


from torch.nn.modules.container import ModuleList


from torch.nn.init import xavier_uniform_


from torch.nn.modules.dropout import Dropout


from torch.nn.modules.linear import Linear


from torch.nn.modules.normalization import LayerNorm


from collections import defaultdict


from torch.utils.data import DataLoader


from torchaudio.transforms import MelSpectrogram


from torchaudio.transforms import AmplitudeToDB


import time


from itertools import permutations


from scipy.optimize import linear_sum_assignment


from numpy import random as nr


from torch.utils.data import WeightedRandomSampler


import random


from functools import partial


from torch.optim.lr_scheduler import _LRScheduler


class EncoderDecoderAttractor(nn.Module):

    def __init__(self, n_units, encoder_dropout=0.1, decoder_dropout=0.1):
        super(EncoderDecoderAttractor, self).__init__()
        self.encoder = nn.LSTM(n_units, n_units, 1, batch_first=True, dropout=encoder_dropout)
        self.decoder = nn.LSTM(n_units, n_units, 1, batch_first=True, dropout=decoder_dropout)
        self.counter = nn.Linear(n_units, 1)
        self.n_units = n_units

    def eda_forward(self, xs, zeros):
        _, (hn, cn) = self.encoder(xs)
        attractors, _ = self.decoder(zeros, (hn, cn))
        return attractors

    def estimate(self, xs, max_n_speakers=15):
        """
        Calculate attractors from embedding sequences
        without prior knowledge of number of speakers

        Args:
        xs:
         
        """
        zeros = torch.zeros((xs.shape[0], max_n_speakers, xs.shape[-1]), dtype=xs.dtype, device=xs.device)
        attractors = self.eda_forward(xs, zeros)
        probs = torch.sigmoid(self.counter(attractors).squeeze(dim=-1))
        return attractors, probs

    def forward(self, xs, n_speakers) ->tuple[Tensor, Tensor]:
        """
        Calculate attractors from embedding sequences with given number of speakers

        Args:
        xs: (B, T, E)
        n_speakers: list of number of speakers in batch
        """
        zeros = torch.zeros((xs.shape[0], max(n_speakers) + 1, xs.shape[-1]), dtype=xs.dtype, device=xs.device)
        attractors = self.eda_forward(xs, zeros)
        labels = torch.cat([torch.tensor([[1] * n_spk + [0]], dtype=torch.float32, device=xs.device) for n_spk in n_speakers], dim=1)
        logit = torch.cat([self.counter(att[:n_spk + 1, :]).reshape(-1, n_spk + 1) for att, n_spk in zip(attractors, n_speakers)], dim=1)
        loss = F.binary_cross_entropy_with_logits(logit, labels)
        return loss, attractors


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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: 'Tensor', max_nspks):
        pe = self.pe[:, :max_nspks, :]
        pe = pe.unsqueeze(dim=0).repeat(x.shape[0], x.shape[1], 1, 1)
        x = x.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1)
        return pe


class TransformerModel(nn.Module):

    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=False):
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
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=False, activation=None):
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
        if activation:
            output = activation(output)
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


class TransformerEDADiarization(nn.Module):

    def __init__(self, n_speakers, in_size, n_units, n_heads, n_layers, dropout, attractor_loss_ratio=1.0, attractor_encoder_dropout=0.1, attractor_decoder_dropout=0.1):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          attractor_loss_ratio (float)
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
        """
        super(TransformerEDADiarization, self).__init__()
        self.n_speakers = n_speakers
        self.enc = TransformerModel(in_size, n_heads, n_units, n_layers, dropout=dropout)
        self.eda = EncoderDecoderAttractor(n_units, encoder_dropout=attractor_encoder_dropout, decoder_dropout=attractor_decoder_dropout)
        self.attractor_loss_ratio = attractor_loss_ratio

    def forward(self, src, tgt, ilens):
        n_speakers = [t.shape[1] for t in tgt]
        emb = self.enc(src)
        attractor_loss, attractors = self.eda(emb, n_speakers)
        output = torch.bmm(emb, attractors.transpose(1, 2))
        output = [out[:ilen, :n_spk] for out, ilen, n_spk in zip(output, ilens, n_speakers)]
        return output, self.attractor_loss_ratio * attractor_loss, emb, attractors[:, :-1, :]

    def test(self, src, ilens, **kwargs):
        n_spk = kwargs.get('n_spk')
        th = kwargs.get('th')
        emb = self.enc(src)
        order = np.arange(emb.shape[1])
        np.random.shuffle(order)
        attractors, probs = self.eda.estimate(emb[:, order, :])
        output = torch.bmm(emb, attractors.transpose(1, 2))
        output_active = []
        for p, y, ilen in zip(probs, output, ilens):
            if n_spk is not None:
                output_active.append(y[:ilen, :n_spk])
            elif th is not None:
                silence = torch.where(p < th)[0]
                n_spk = silence[0] if silence.size else None
                output_active.append(y[:ilen, :n_spk])
        return output_active, emb, attractors[:, :-1, :]


def _get_activation_fn(activation: 'str') ->Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))


class TransformerEncoderFusionLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: 'int', nhead: 'int', dim_feedforward: 'int'=2048, dropout: 'float'=0.1, activation: 'Union[str, Callable[[Tensor], Tensor]]'=F.relu, layer_norm_eps: 'float'=1e-05, batch_first: 'bool'=False, norm_first: 'bool'=False, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderFusionLayer, self).__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.norm11 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm12 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm21 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm22 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout11 = Dropout(dropout)
        self.dropout21 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderFusionLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: 'Tensor', src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None) ->Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError('only bool and floating types of key_padding_mask are supported')
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f'input not batched; expected src.dim() of 3 but got {src.dim()}'
        elif self.training:
            why_not_sparsity_fast_path = 'training is enabled'
        elif not self.self_attn1.batch_first:
            why_not_sparsity_fast_path = 'self_attn.batch_first was not True'
        elif not self.self_attn1._qkv_same_embed_dim:
            why_not_sparsity_fast_path = 'self_attn._qkv_same_embed_dim was not True'
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = 'activation_relu_or_gelu was not True'
        elif not self.norm11.eps == self.norm12.eps:
            why_not_sparsity_fast_path = 'norm1.eps is not equal to norm2.eps'
        elif src_mask is not None:
            why_not_sparsity_fast_path = 'src_mask is not supported for fastpath'
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = 'src_key_padding_mask is not supported with NestedTensor input for fastpath'
        elif self.self_attn1.num_heads % 2 == 1:
            why_not_sparsity_fast_path = 'num_head is odd'
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = 'autocast is enabled'
        if not why_not_sparsity_fast_path:
            tensor_args = src, self.self_attn1.in_proj_weight, self.self_attn1.in_proj_bias, self.self_attn1.out_proj.weight, self.self_attn1.out_proj.bias, self.self_attn2.in_proj_weight, self.self_attn2.in_proj_bias, self.self_attn2.out_proj.weight, self.self_attn2.out_proj.bias, self.norm11.weight, self.norm11.bias, self.norm12.weight, self.norm12.bias, self.norm21.weight, self.norm21.bias, self.norm22.weight, self.norm22.bias, self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = 'some Tensor argument has_torch_function'
            elif not all(x.is_cuda or 'cpu' in str(x.device) for x in tensor_args):
                why_not_sparsity_fast_path = 'some Tensor argument is neither CUDA nor CPU'
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = 'grad is enabled and at least one of query or the input/output projection weights or biases requires_grad'
            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(src, self.self_attn1.embed_dim, self.self_attn1.num_heads, self.self_attn1.in_proj_weight, self.self_attn1.in_proj_bias, self.self_attn1.out_proj.weight, self.self_attn1.out_proj.bias, self.self_attn2.embed_dim, self.self_attn2.num_heads, self.self_attn2.in_proj_weight, self.self_attn2.in_proj_bias, self.self_attn2.out_proj.weight, self.self_attn2.out_proj.bias, self.activation_relu_or_gelu == 2, self.norm_first, self.norm11.eps, self.norm11.weight, self.norm11.bias, self.norm12.weight, self.norm12.bias, self.norm21.eps, self.norm21.weight, self.norm21.bias, self.norm22.weight, self.norm22.bias, self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias, src_mask if src_mask is not None else src_key_padding_mask, 1 if src_key_padding_mask is not None else 0 if src_mask is not None else None)
        B, T, C, D = src.shape
        x = src.transpose(1, 2).reshape(B * C, T, D)
        if self.norm_first:
            x = x + self._sa_block1(self.norm11(x), src_mask, src_key_padding_mask)
        else:
            x = self.norm11(x + self._sa_block1(x, src_mask, src_key_padding_mask))
        x = x.reshape(B, C, T, D).transpose(1, 2).reshape(B * T, C, D)
        if self.norm_first:
            x = x + self._sa_block2(self.norm21(x), None, None)
            x = x + self._ff_block(self.norm22(x))
        else:
            x = self.norm21(x + self._sa_block2(x, None, None))
            x = self.norm22(x + self._ff_block(x))
        x = x.reshape(B, T, C, D)
        return x

    def _sa_block1(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x = self.self_attn1(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout11(x)

    def _sa_block2(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x = self.self_attn2(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout21(x)

    def _ff_block(self, x: 'Tensor') ->Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class MaskedTransformerDecoderModel(nn.Module):

    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward, dropout=0.5, has_mask=False, max_seqlen=500, has_pos=False, mask_delay=0):
        super(MaskedTransformerDecoderModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos
        self.has_mask = has_mask
        self.max_seqlen = max_seqlen
        self.mask_delay = mask_delay
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        self.pos_enc = PositionalEncoding(n_units, dropout)
        self.convert = nn.Linear(n_units * 2, n_units)
        decoder_layers = TransformerEncoderFusionLayer(n_units, n_heads, dim_feedforward, dropout, batch_first=True)
        self.attractor_decoder = TransformerEncoder(decoder_layers, n_layers)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=-self.mask_delay) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, emb: 'Tensor', max_nspks: 'int', activation: 'Optional[Callable]'=None):
        pos_enc = self.pos_enc(emb, max_nspks)
        attractors_init: 'Tensor' = self.convert(torch.cat([emb.unsqueeze(dim=2).repeat(1, 1, max_nspks, 1), pos_enc], dim=-1))
        t_mask = self._generate_square_subsequent_mask(emb.shape[1], emb.device)
        attractors = self.attractor_decoder(attractors_init, t_mask)
        return attractors


class MaskedTransformerEncoderModel(nn.Module):

    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_mask=False, max_seqlen=500, has_pos=False, mask_delay=0):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(MaskedTransformerEncoderModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos
        self.has_mask = has_mask
        self.max_seqlen = max_seqlen
        self.mask_delay = mask_delay
        self.bn = nn.BatchNorm1d(in_size)
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=-self.mask_delay) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, activation=None):
        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)
        src = self.bn(src.transpose(1, 2)).transpose(1, 2)
        src_mask = None
        if self.has_mask:
            src_mask = self._generate_square_subsequent_mask(src.shape[1], src.device)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        src = src.transpose(0, 1)
        if self.has_pos:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.transpose(0, 1)
        if activation:
            output = activation(output)
        return output


class OnlineTransformerDADiarization(nn.Module):

    def __init__(self, n_speakers, in_size, n_units, n_heads, enc_n_layers, dec_n_layers, dropout, has_mask, max_seqlen, dec_dim_feedforward, conv_delay=9, mask_delay=0, decom_kernel_size=64):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(OnlineTransformerDADiarization, self).__init__()
        self.n_speakers = n_speakers
        self.delay = conv_delay
        self.enc = MaskedTransformerEncoderModel(in_size, n_heads, n_units, enc_n_layers, dropout=dropout, has_mask=has_mask, max_seqlen=max_seqlen, mask_delay=mask_delay)
        self.dec = MaskedTransformerDecoderModel(in_size, n_heads, n_units, dec_n_layers, dim_feedforward=dec_dim_feedforward, dropout=dropout, has_mask=has_mask, max_seqlen=max_seqlen, mask_delay=mask_delay)
        self.cnn = nn.Conv1d(n_units, n_units, kernel_size=2 * conv_delay + 1, padding=9)

    def forward(self, src, tgt, ilens):
        n_speakers = [t.shape[1] for t in tgt]
        max_nspks = max(n_speakers)
        emb = self.enc(src)
        B, T, D = emb.shape
        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        emb: 'Tensor' = self.cnn(emb.transpose(1, 2)).transpose(1, 2)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        attractors = self.dec(emb, max_nspks)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True)
        attn_map = emb.matmul(emb.transpose(-1, -2))
        attn_norm = torch.norm(emb, dim=-1, keepdim=True)
        attn_norm = attn_norm.matmul(attn_norm.transpose(-1, -2))
        attn_map = attn_map / (attn_norm + 1e-06)
        tgt_pad = [F.pad(t, (0, max_nspks - t.shape[1]), 'constant', 0) for t in tgt]
        tgt_pad = nn.utils.rnn.pad_sequence(tgt_pad, padding_value=0, batch_first=True)
        label_map = tgt_pad.matmul(tgt_pad.transpose(-1, -2))
        tgt_norm = torch.norm(tgt_pad, dim=-1, keepdim=True)
        tgt_norm = tgt_norm.matmul(tgt_norm.transpose(-1, -2))
        label_map = label_map / (tgt_norm + 1e-06)
        emb_consis_loss = F.mse_loss(attn_map, label_map)
        output = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        output = [out[:ilen, :n_spk] for out, ilen, n_spk in zip(output, ilens, n_speakers)]
        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        attractors = [attr[:ilen, 1:n_spk] for attr, ilen, n_spk in zip(attractors, ilens, n_speakers)]
        return output, emb_consis_loss, emb, attractors

    def test(self, src, ilens, max_nspks=6):
        emb = self.enc(src)
        B, T, D = emb.shape
        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        emb = nn.utils.rnn.pad_sequence(emb, padding_value=0, batch_first=True)
        emb: 'Tensor' = self.cnn(emb.transpose(1, 2)).transpose(1, 2)
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        attractors = self.dec(emb, max_nspks)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True)
        output = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        output = [out[:ilen] for out, ilen in zip(output, ilens)]
        emb = [e[:ilen] for e, ilen in zip(emb, ilens)]
        attractors = [attr[:ilen] for attr, ilen in zip(attractors, ilens)]
        return output, emb, attractors


class TransformerEncoderLayerC(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: 'int', nhead: 'int', dim_feedforward: 'int'=2048, dropout: 'float'=0.1, activation: 'Union[str, Callable[[Tensor], Tensor]]'=F.relu, layer_norm_eps: 'float'=1e-05, batch_first: 'bool'=False, norm_first: 'bool'=False, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayerC, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayerC, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: 'Tensor', src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None) ->Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError('only bool and floating types of key_padding_mask are supported')
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f'input not batched; expected src.dim() of 3 but got {src.dim()}'
        elif self.training:
            why_not_sparsity_fast_path = 'training is enabled'
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = 'self_attn.batch_first was not True'
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = 'self_attn._qkv_same_embed_dim was not True'
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = 'activation_relu_or_gelu was not True'
        elif not self.norm1.eps == self.norm2.eps:
            why_not_sparsity_fast_path = 'norm1.eps is not equal to norm2.eps'
        elif src_mask is not None:
            why_not_sparsity_fast_path = 'src_mask is not supported for fastpath'
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = 'src_key_padding_mask is not supported with NestedTensor input for fastpath'
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = 'num_head is odd'
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = 'autocast is enabled'
        if not why_not_sparsity_fast_path:
            tensor_args = src, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias, self.norm1.weight, self.norm1.bias, self.norm2.weight, self.norm2.bias, self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = 'some Tensor argument has_torch_function'
            elif not all(x.is_cuda or 'cpu' in str(x.device) for x in tensor_args):
                why_not_sparsity_fast_path = 'some Tensor argument is neither CUDA nor CPU'
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = 'grad is enabled and at least one of query or the input/output projection weights or biases requires_grad'
            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(src, self.self_attn.embed_dim, self.self_attn.num_heads, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias, self.activation_relu_or_gelu == 2, self.norm_first, self.norm1.eps, self.norm1.weight, self.norm1.bias, self.norm2.weight, self.norm2.bias, self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias, src_mask if src_mask is not None else src_key_padding_mask, 1 if src_key_padding_mask is not None else 0 if src_mask is not None else None)
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: 'Tensor') ->Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class IncrementalSelfAttention(nn.Module):

    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, query, key, value, past_key=None, past_value=None):
        """
        Args:
            query: (batch_size, 1, d_model)
            key: (batch_size, 1, d_model)
            value: (batch_size, 1, d_model)
            past_key: (batch_size, seq_len, d_model)
            past_value: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, 1, d_model)
            new_key: (batch_size, seq_len+1, d_model)
            new_value: (batch_size, seq_len+1, d_model)
        """
        if past_key is not None and past_value is not None:
            comb_key = torch.cat([past_key, key], dim=1)
            comb_value = torch.cat([past_value, value], dim=1)
        else:
            comb_key = key
            comb_value = value
        attn_output, _ = self.attention(query, comb_key, comb_value)
        return attn_output, comb_key, comb_value


class StreamingTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=None):
        super().__init__()
        self.self_attn = IncrementalSelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation if activation else nn.ReLU()

    def forward(self, x, past_key=None, past_value=None):
        """
        Args:
            x: (batch_size, 1, d_model)
            past_key: (batch_size, seq_len, d_model)
            past_value: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, 1, d_model)
            new_key: (batch_size, seq_len+1, d_model)
            new_value: (batch_size, seq_len+1, d_model)
        """
        query = x
        key, value = x, x
        attn_output, new_key, new_value = self._sa_block(query, key, value, past_key, past_value)
        attn_output = self.norm1(attn_output + x)
        ff_output = self._ff_block(attn_output)
        output = self.norm2(ff_output + attn_output)
        return output, new_key, new_value

    def _ff_block(self, x: 'Tensor') ->Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _sa_block(self, query: 'Tensor', key, value, past_key=None, past_value=None) ->Tensor:
        x, new_key, new_value = self.self_attn(query, key, value, past_key, past_value)
        return self.dropout1(x), new_key, new_value


class StreamingTransformerEncoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList([StreamingTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation) for _ in range(num_layers)])
        self.cache = [{} for _ in range(num_layers)]

    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, d_model)
        Returns:
            output: (batch_size, 1, d_model)
        """
        for i, layer in enumerate(self.layers):
            cache = self.cache[i]
            past_key = cache.get('key', None)
            past_value = cache.get('value', None)
            x, new_key, new_value = layer(x, past_key, past_value)
            self.cache[i]['key'] = new_key.detach()
            self.cache[i]['value'] = new_value.detach()
        return x


class PITLoss(nn.Module):

    def __init__(self, n_spks=4) ->None:
        super(PITLoss, self).__init__()
        self.n_spks = n_spks
        trav_idx = []
        for x in permutations(list(range(n_spks))):
            trav_idx += x
        self.trav_idx = torch.tensor(trav_idx).long()

    def forward(self, preds, labels):
        B, T, _ = preds.shape
        labels_all_case = labels[:, :, self.trav_idx].reshape(B, T, -1, self.n_spks)
        case_num = labels_all_case.shape[-2]
        preds_all_case = preds.unsqueeze(-1).repeat(1, 1, 1, case_num).transpose(-1, -2)
        loss_all_case = F.binary_cross_entropy_with_logits(preds_all_case, labels_all_case, reduction='none').mean(-1).mean(1)
        min_idx = torch.argmin(loss_all_case, dim=-1)
        selected_loss = loss_all_case[torch.arange(B), min_idx]
        pit_min_loss = selected_loss.sum() / B
        perm_labels = labels_all_case[torch.arange(B), :, min_idx, :]
        return pit_min_loss, perm_labels

    def swap(self, a, i, j):
        temp = a[i]
        a[i] = a[j]
        a[j] = temp

    def dfs(self, a, depth=0):
        r = []
        if depth == len(a):
            r += a
        for i in range(depth, len(a)):
            self.swap(a, i, depth)
            r += self.dfs(a, depth + 1)
            self.swap(a, depth, i)
        return r


class ContextBuilder(torch.nn.Module):

    def __init__(self, context_size) ->None:
        super(ContextBuilder, self).__init__()
        self.context_size = context_size

    def forward(self, x):
        bsz, T, _ = x.shape
        x_pad = torch.nn.functional.pad(x, (0, 0, self.context_size, self.context_size))
        return x_pad.unfold(1, T, 1).reshape(bsz, -1, T).transpose(-1, -2)


class TorchScaler(torch.nn.Module):
    """
    This torch module implements scaling for input tensors, both instance based
    and dataset-wide statistic based.

    Args:
        statistic: str, (default='dataset'), represent how to compute the statistic for normalisation.
            Choice in {'dataset', 'instance'}.
             'dataset' needs to be 'fit()' with a dataloader of the dataset.
             'instance' apply the normalisation at an instance-level, so compute the statitics on the instance
             specified, it can be a clip or a batch.
        normtype: str, (default='standard') the type of normalisation to use.
            Choice in {'standard', 'mean', 'minmax'}. 'standard' applies a classic normalisation with mean and standard
            deviation. 'mean' substract the mean to the data. 'minmax' substract the minimum of the data and divide by
            the difference between max and min.
    """

    def __init__(self, statistic='dataset', normtype='standard', dims=(1, 2), eps=1e-08):
        super(TorchScaler, self).__init__()
        assert statistic in ['dataset', 'instance']
        assert normtype in ['standard', 'mean', 'minmax']
        if statistic == 'dataset' and normtype == 'minmax':
            raise NotImplementedError('statistic==dataset and normtype==minmax is not currently implemented.')
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == 'dataset':
            super(TorchScaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.statistic == 'dataset':
            super(TorchScaler, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def fit(self, dataloader, transform_func=lambda x: x[0]):
        """
        Scaler fitting

        Args:
            dataloader (DataLoader): training data DataLoader
            transform_func (lambda function, optional): Transforms applied to the data.
                Defaults to lambdax:x[0].
        """
        indx = 0
        for batch in tqdm.tqdm(dataloader):
            feats = transform_func(batch)
            if indx == 0:
                mean = torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared = torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
            else:
                mean += torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared += torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
            indx += 1
        mean /= indx
        mean_squared /= indx
        self.register_buffer('mean', mean)
        self.register_buffer('mean_squared', mean_squared)

    def forward(self, tensor):
        if self.statistic == 'dataset':
            assert hasattr(self, 'mean') and hasattr(self, 'mean_squared'), 'TorchScaler should be fit before used if statistics=dataset'
            assert tensor.ndim == self.mean.ndim, 'Pre-computed statistics '
            if self.normtype == 'mean':
                return tensor - self.mean
            elif self.normtype == 'standard':
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (tensor - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError
        elif self.normtype == 'mean':
            return tensor - torch.mean(tensor, self.dims, keepdim=True)
        elif self.normtype == 'standard':
            return (tensor - torch.mean(tensor, self.dims, keepdim=True)) / (torch.std(tensor, self.dims, keepdim=True) + self.eps)
        elif self.normtype == 'minmax':
            return (tensor - torch.amin(tensor, dim=self.dims, keepdim=True)) / (torch.amax(tensor, dim=self.dims, keepdim=True) - torch.amin(tensor, dim=self.dims, keepdim=True) + self.eps)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContextBuilder,
     lambda: ([], {'context_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (EncoderDecoderAttractor,
     lambda: ([], {'n_units': 4}),
     lambda: ([torch.rand([4, 4, 4]), [4, 4]], {}),
     True),
    (IncrementalSelfAttention,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (MaskedTransformerEncoderModel,
     lambda: ([], {'in_size': 4, 'n_heads': 4, 'n_units': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (PITLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), 0], {}),
     True),
    (StreamingTransformerEncoder,
     lambda: ([], {'d_model': 4, 'nhead': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (StreamingTransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (TransformerEDADiarization,
     lambda: ([], {'n_speakers': 4, 'in_size': 4, 'n_units': 4, 'n_heads': 4, 'n_layers': 1, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), [4, 4]], {}),
     False),
    (TransformerEncoderFusionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerEncoderLayerC,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (TransformerModel,
     lambda: ([], {'in_size': 4, 'n_heads': 4, 'n_units': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_Audio_WestlakeU_FS_EEND(_paritybench_base):
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

