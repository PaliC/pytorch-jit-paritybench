import sys
_module = sys.modules[__name__]
del sys
length_adaptive_transformer = _module
drop_and_restore_utils = _module
evolution = _module
modeling_bert = _module
modeling_distilbert = _module
modeling_utils = _module
trainer = _module
training_args = _module
run_glue = _module
run_squad = _module

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


import logging


import random


import numpy as np


import torch


from torch import nn


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import torch.nn as nn


import warnings


from collections import defaultdict


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import torch.nn.functional as F


from torch.nn import KLDivLoss


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataset import Dataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


def expand_gather(input, dim, index):
    size = list(input.size())
    size[dim] = -1
    return input.gather(dim, index.expand(*size))


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f'{self} should be used as a decoder model if cross attention is added'
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False, output_length=None, always_keep_cls_token=True):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(self, 'crossattention'), f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`'
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        if output_length is not None:
            assert output_attentions
            attention_probs = self_attention_outputs[1]
            significance_score = attention_probs.sum(2).sum(1)
            if always_keep_cls_token:
                keep_indices = significance_score[:, 1:].topk(output_length - 1, 1)[1] + 1
                cls_index = keep_indices.new_zeros((keep_indices.size(0), 1))
                keep_indices = torch.cat((cls_index, keep_indices), 1)
            else:
                keep_indices = significance_score.topk(output_length, 1)[1]
            attention_output = expand_gather(attention_output, 1, keep_indices.unsqueeze(-1))
        else:
            keep_indices = None
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        return outputs, keep_indices

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False, layer_config=None, length_config=None, always_keep_cls_token=True):
        bsz, tsz, dim = hidden_states.size()
        if length_config is not None:
            restored_hidden_states = hidden_states
            remain_indices = torch.arange(tsz, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if layer_config is not None and i not in layer_config:
                continue
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_output_length = length_config[i] if length_config is not None else None
            if getattr(self.config, 'gradient_checkpointing', False):

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, layer_output_length, always_keep_cls_token)
                    return custom_forward
                layer_outputs, keep_indices = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs, keep_indices = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions, output_length=layer_output_length, always_keep_cls_token=always_keep_cls_token)
            hidden_states = layer_outputs[0]
            if layer_output_length:
                remain_indices = remain_indices.gather(1, keep_indices)
                restored_hidden_states = restored_hidden_states.scatter(1, remain_indices.unsqueeze(-1).expand(-1, -1, dim), hidden_states)
                if attention_mask is not None:
                    attention_mask = expand_gather(attention_mask, 3, keep_indices.unsqueeze(1).unsqueeze(2))
                    if attention_mask.size(2) > 1:
                        attention_mask = expand_gather(attention_mask, 2, keep_indices.unsqueeze(1).unsqueeze(3))
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        last_hidden_state = restored_hidden_states if length_config is not None else hidden_states
        if not return_dict:
            return tuple(v for v in [last_hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=all_hidden_states, attentions=all_attentions)


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_heads == 0
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_length=None, always_keep_cls_token=True):
        """
        Parameters
        ----------
        hidden_states: torch.tensor(bs, seq_length, dim)
        attention_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        attention_probs: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        layer_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        self_attention_outputs = self.attention(query=hidden_states, key=hidden_states, value=hidden_states, mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions)
        if output_attentions:
            attention_output, attention_probs = self_attention_outputs
        else:
            assert type(self_attention_outputs) == tuple
            attention_output = self_attention_outputs[0]
        attention_output = self.sa_layer_norm(attention_output + hidden_states)
        if output_length is not None:
            assert output_attentions
            significance_score = attention_probs.sum(2).sum(1)
            if always_keep_cls_token:
                keep_indices = significance_score[:, 1:].topk(output_length - 1, 1)[1] + 1
                cls_index = keep_indices.new_zeros((keep_indices.size(0), 1))
                keep_indices = torch.cat((cls_index, keep_indices), 1)
            else:
                keep_indices = significance_score.topk(output_length, 1)[1]
            attention_output = expand_gather(attention_output, 1, keep_indices.unsqueeze(-1))
        else:
            keep_indices = None
        layer_output = self.ffn(attention_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        output = layer_output,
        if output_attentions:
            output = (attention_probs,) + output
        return output, keep_indices


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = config.n_layers
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=None, layer_config=None, length_config=None, always_keep_cls_token=True):
        """
        Parameters
        ----------
        hidden_states: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attention_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        bsz, tsz, dim = hidden_states.size()
        if length_config is not None:
            restored_hidden_states = hidden_states
            remain_indices = torch.arange(tsz, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if layer_config is not None and i not in layer_config:
                continue
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_output_length = length_config[i] if length_config is not None else None
            layer_outputs, keep_indices = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions, output_length=layer_output_length, always_keep_cls_token=always_keep_cls_token)
            hidden_states = layer_outputs[-1]
            if layer_output_length:
                remain_indices = remain_indices.gather(1, keep_indices)
                restored_hidden_states = restored_hidden_states.scatter(1, remain_indices.unsqueeze(-1).expand(-1, -1, dim), hidden_states)
                if attention_mask is not None:
                    attention_mask = attention_mask.gather(1, keep_indices)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1
        last_hidden_state = restored_hidden_states if length_config is not None else hidden_states
        if not return_dict:
            return tuple(v for v in [last_hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=all_hidden_states, attentions=all_attentions)

