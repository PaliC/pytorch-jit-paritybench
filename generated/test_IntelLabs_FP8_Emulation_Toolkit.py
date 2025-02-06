import sys
_module = sys.modules[__name__]
del sys
modeling_bert = _module
run_squad = _module
imagenet_qat = _module
imagenet_test = _module
launch = _module
test_scaleshift = _module
train = _module
utils = _module
modeling_bert = _module
run_qa_beam_search_no_trainer = _module
run_qa_no_trainer = _module
utils_qa = _module
lr_scheduler = _module
main_amp = _module
main_amp_cpu = _module
mpemu = _module
bfloat16_emu = _module
cmodel = _module
simple = _module
conv_grad_test = _module
conv_test = _module
gemm_grad_test = _module
gemm_irregular_test = _module
gemm_test = _module
linear_test = _module
net = _module
e3m4_emu = _module
e4m3_emu = _module
e5m2_emu = _module
hybrid_emu = _module
module_wrappers = _module
adasparse = _module
aggregate = _module
eltwise = _module
matmul = _module
mpt_emu = _module
pytquant = _module
cpp = _module
fpemu = _module
cuda = _module
fpemu = _module
hip = _module
fpemu = _module
test = _module
qutils = _module
scale_shift = _module
sparse_utils = _module
stats_collector = _module
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


import math


import warnings


from typing import Optional


from typing import Tuple


import torch


import torch.utils.checkpoint


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import logging


import random


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


import time


import copy


import collections


import torch.utils.data


import torchvision


from torchvision import transforms


import torch.quantization


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from collections import defaultdict


from collections import deque


import torch.distributed as dist


from math import pi


from math import cos


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data.distributed


import torchvision.models as models


from torch.nn.parallel import DistributedDataParallel as DDP


from collections import OrderedDict


from torch.quantization.fuser_method_mappings import fuse_conv_bn


from torch.quantization.fuser_method_mappings import fuse_conv_bn_relu


from torch.autograd import Function


import torch.nn.functional as F


import torch.optim as optim


import numpy


from enum import Enum


from scipy.optimize import root_scalar


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse('1.6.0'):
            self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device), persistent=False)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attn_scores_matmul = module_wrappers.Matmul()
        self.context_matmul = module_wrappers.Matmul()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.is_decoder:
            past_key_value = key_layer, value_layer
        attention_scores = self.attn_scores_matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = self.context_matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.eltwise_add = module_wrappers.EltwiseAdd()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(self.eltwise_add(hidden_states, input_tensor))
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.eltwise_add = module_wrappers.EltwiseAdd()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(self.eltwise_add(hidden_states, input_tensor))
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


logger = logging.getLogger(__name__)


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += layer_outputs[-1],
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class WeightMaskStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input > 0.0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2 - 4 * torch.abs(input)
        additional[zero_index] = 0.0
        additional[middle_index] = 0.4
        return grad_input * additional


class SparseLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.threshold = nn.Parameter(torch.Tensor(out_features))
        self.weight_mask = WeightMaskStep.apply
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0)

    def forward(self, input):
        abs_weight = torch.abs(self.weight)
        threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight - threshold
        mask = self.weight_mask(abs_weight)
        ratio = torch.sum(mask) / mask.numel()
        if ratio <= 0.01:
            with torch.no_grad():
                self.threshold.data.fill_(0)
            abs_weight = torch.abs(self.weight)
            threshold = self.threshold.view(abs_weight.shape[0], -1)
            abs_weight = abs_weight - threshold
            mask = self.weight_mask(abs_weight)
        masked_weight = self.weight * mask
        output = torch.nn.functional.linear(input, masked_weight, self.bias)
        return output

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class SparseConv2d(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(torch.Tensor(out_c, in_c // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.threshold = nn.Parameter(torch.Tensor(out_c))
        self.weight_mask = WeightMaskStep.apply
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.0)

    def forward(self, x):
        weight_shape = self.weight.shape
        threshold = self.threshold.view(weight_shape[0], -1)
        weight = torch.abs(self.weight)
        weight = weight.view(weight_shape[0], -1)
        weight = weight - threshold
        mask = self.weight_mask(weight)
        mask = mask.view(weight_shape)
        ratio = torch.sum(mask) / mask.numel()
        if ratio <= 0.01:
            with torch.no_grad():
                self.threshold.data.fill_(0.0)
            threshold = self.threshold.view(weight_shape[0], -1)
            weight = torch.abs(self.weight)
            weight = weight.view(weight_shape[0], -1)
            weight = weight - threshold
            mask = self.weight_mask(weight)
            mask = mask.view(weight_shape)
        masked_weight = self.weight * mask
        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(SparseConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Norm(nn.Module):

    def __init__(self, p='fro', dim=None, keepdim=False):
        super(Norm, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: 'torch.Tensor'):
        return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self) ->str:
        return 'p={}, dim={}, keepdim: {}'.format(self.p, self.dim, self.keepdim)


class Mean(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Mean, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: 'torch.Tensor'):
        return torch.mean(x, *self.args, **self.kwargs)

    def extra_repr(self) ->str:
        return 'args={}, kwargs={}'.format(self.args, self.kwargs)


class EltwiseAdd(nn.Module):

    def __init__(self, inplace=False):
        super(EltwiseAdd, self).__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res

    def extra_repr(self) ->str:
        return 'inplace={}'.format(self.inplace)


class EltwiseMul(nn.Module):

    def __init__(self, inplace=False):
        super(EltwiseMult, self).__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res

    def extra_repr(self) ->str:
        return 'inplace={}'.format(self.inplace)


class EltwiseDiv(nn.Module):

    def __init__(self, inplace=False):
        super(EltwiseDiv, self).__init__()
        self.inplace = inplace

    def forward(self, x: 'torch.Tensor', y):
        if self.inplace:
            return x.div_(y)
        return x.div(y)

    def extra_repr(self) ->str:
        return 'inplace={}'.format(self.inplace)


class Matmul(nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, a: 'torch.Tensor', b: 'torch.Tensor'):
        return torch.matmul(a, b)


class BatchMatmul(nn.Module):

    def __init__(self):
        super(BatchMatmul, self).__init__()

    def forward(self, a: 'torch.Tensor', b: 'torch.Tensor'):
        return torch.bmm(a, b)


class AddMatmul(nn.Module):

    def __init__(self):
        super(AddMatmul, self).__init__()

    def forward(self, input: 'torch.Tensor', mat1: 'torch.Tensor', mat2: 'torch.Tensor', beta=1, alpha=1):
        return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


class ScaleShift(torch.nn.Module):

    def __init__(self, num_features):
        super(ScaleShift, self).__init__()
        self.num_features = num_features
        self.weight = torch.nn.Parameter(torch.Tensor(num_features))
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        input_t = input.transpose(1, -1)
        input_t = input_t * self.weight + self.bias
        output = input_t.transpose(1, -1)
        return output

    def __repr__(self):
        return 'ScaleShift({})'.format(self.num_features)

    @staticmethod
    def generate_from_batchnorm(module: 'torch.nn.BatchNorm2d'):
        """
            Helper function for converting Batchnorm2d to ScaleShift
        """
        bn_state_dict = module.state_dict()
        num_features = module.num_features
        eps = module.eps
        rmean = bn_state_dict['running_mean']
        rvar = bn_state_dict['running_var']
        gamma = bn_state_dict['weight']
        beta = bn_state_dict['bias']
        ss_module = ScaleShift(num_features)
        with torch.no_grad():
            denom = torch.sqrt(rvar + eps)
            scale = gamma.div(denom)
            shift = beta - gamma.mul(rmean).div(denom)
            ss_module.weight.copy_(scale)
            ss_module.bias.copy_(shift)
        return ss_module


class TensorDump(torch.nn.Module):

    def __init__(self, name=''):
        self.name = name
        self.tensors = []

    def forward(self, tensor):
        self.tensors.append(tensor)

    def dump(self):
        import numpy as np
        pickle.dump(np.array([tensor.detach().numpy() for tensor in self.tensors]), open(self.name + '.pickle', 'wb'))


class TensorDumpListWrapper(torch.nn.Module):
    """docstring for MinMaxStats of a tensor"""

    def __init__(self, name=''):
        super(TensorDumpListWrapper, self).__init__()
        self.initiated = False
        self.tupled_input = False
        self.tensor_dump_list = []
        self.name = name

    def forward(self, input):
        if not self.initiated:
            if type(input) == tuple:
                self.tupled_input = True
                for i in range(len(input)):
                    self.tensor_dump_list.append(TensorDump(name=self.name + '_{}'.format(i)))
            else:
                self.tensor_dump_list = [TensorDump(name=self.name + '_0')]
            self.initiated = True
        if self.tupled_input:
            for i in range(len(input)):
                self.tensor_dump_list[i].forward(input[i])
        else:
            self.tensor_dump_list[0].forward(input)

    def dump(self):
        for tensor_dump in self.tensor_dump_list:
            tensor_dump.dump()


class ArchiveStats(torch.nn.Module):

    def __init__(self):
        self.tensors = []

    def forward(self, tensor):
        self.tensors.append(tensor)


class TensorFullIntQuantParams(object):
    """
    min_val : float
    max_val : float
    qconfig : TensorQuantConfig
    """

    def __init__(self, min_val, max_val, qconfig):
        super(TensorFullIntQuantParams, self).__init__()
        self.qconfig = qconfig
        self.min_val, self.max_val, self.scale, self.zero_point = self._calculate_int8_qparams_base(qconfig.dtype, qconfig.scheme, min_val, max_val)

    def quantize(self, tensor_f):
        tensor_int = torch.round(tensor_f / self.scale + self.zero_point)
        min_int = -128
        max_int = 127
        if self.qconfig.dtype == 'uint8':
            min_int = 0
            max_int = 255
        tensor_int = torch.clamp(tensor_int, min_int, max_int)
        return tensor_int

    def dequantize(self, tensor_int):
        tensor_f = (tensor_int - self.zero_point) * self.scale
        return tensor_f

    def quant_dequant(self, tensor_f):
        return self.dequantize(self.quantize(tensor_f))

    def __repr__(self):
        return '{} Quantization range [{:.4f},{:.4f}] '.format(self.qconfig, self.min_val, self.max_val)

    @staticmethod
    def _calculate_int8_qparams_base(dtype, scheme, min_val, max_val):
        """
        Adapted from https://github.com/pytorch/pytorch/blob/8074779328fa471f484fb74cc6c50d95392fe2c2/torch/quantization/observer.py#L193
        """
        assert min_val <= max_val, 'Minimum value {} has to be less than Maximum value {}'.format(min_val, max_val)
        eps = torch.finfo(torch.float32).eps
        if dtype == 'uint8':
            qmin = 0
            qmax = 255
        elif dtype == 'int8':
            qmin = -128
            qmax = 127
        min_val, max_val = float(min_val), float(max_val)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)
        if min_val == max_val:
            scale = 1.0
            zero_point = 0
        elif scheme == 'sym_full' or scheme == 'sym_channel':
            max_val = max(-min_val, max_val)
            scale = max_val / ((qmax - qmin) / 2)
            scale = max(scale, eps)
            zero_point = 0 if dtype == 'int8' else 128
            min_val = -1 * max_val
        elif scheme == 'asym_full' or scheme == 'asym_channel':
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = max(scale, eps)
            zero_point = qmin - round(min_val / scale)
            zero_point = max(qmin, zero_point)
            zero_point = min(qmax, zero_point)
            zero_point = int(zero_point)
        return min_val, max_val, scale, zero_point


class MinMaxStats(torch.nn.Module):
    """docstring for MinMaxStats of a tensor"""

    def __init__(self, archive_tensors=False):
        super(MinMaxStats, self).__init__()
        self.min_val = None
        self.max_val = None
        self.archive_tensors = archive_tensors
        self.tensors = None

    def forward(self, tensor):
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()
        if self.min_val == None:
            self.min_val = min_val
        elif min_val < self.min_val:
            self.min_val = min_val
        if self.max_val == None:
            self.max_val = max_val
        elif max_val > self.max_val:
            self.max_val = max_val
        if self.archive_tensors:
            if self.tensors is None:
                self.tensors = [tensor]
            else:
                self.tensors.append(tensor)

    def get_tensor_quant_params(self, ten_qconfig):
        assert ten_qconfig.dtype in ['uint8', 'int8']
        ten_qparams = TensorFullIntQuantParams(self.min_val, self.max_val, ten_qconfig)
        return ten_qparams

    def print(self, name=''):
        None


class RunningMinMaxStats(torch.nn.Module):
    """docstring for MinMaxStats of a tensor"""

    def __init__(self, archive_tensors=False):
        super(RunningMinMaxStats, self).__init__()
        self.min_val = None
        self.max_val = None
        self.running_min_val = None
        self.running_max_val = None
        self.running_steps = 0
        self.archive_tensors = archive_tensors
        self.tensors = None

    def forward(self, tensor):
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()
        if self.min_val == None:
            self.min_val = min_val
        elif min_val < self.min_val:
            self.min_val = min_val
        if self.max_val == None:
            self.max_val = max_val
        elif max_val > self.max_val:
            self.max_val = max_val
        if self.running_min_val == None:
            self.running_min_val = min_val
        else:
            self.running_min_val = (self.running_min_val * self.running_steps + min_val) / (self.running_steps + 1)
        if self.running_max_val == None:
            self.running_max_val = max_val
        else:
            self.running_max_val = (self.running_max_val * self.running_steps + max_val) / (self.running_steps + 1)
        self.running_steps += 1
        if self.archive_tensors:
            if self.tensors is None:
                self.tensors = [tensor]
            else:
                self.tensors.append(tensor)

    def get_tensor_quant_params(self, ten_qconfig):
        assert ten_qconfig.dtype in ['uint8', 'int8']
        ten_qparams = TensorFullIntQuantParams(self.running_min_val, self.running_max_val, ten_qconfig)
        return ten_qparams

    def print(self, name=''):
        None


class StatsListWrapper(torch.nn.Module):
    """docstring for MinMaxStats of a tensor"""

    def __init__(self, stats_class, archive_tensors):
        super(StatsListWrapper, self).__init__()
        self.initiated = False
        self.tupled_input = False
        self.stats_list = []
        self.stats_class = stats_class
        self.archive_tensors = archive_tensors

    def forward(self, input):
        if not self.initiated:
            if type(input) == tuple:
                self.tupled_input = True
                for i in range(len(input)):
                    self.stats_list.append(self.stats_class(archive_tensors=self.archive_tensors))
            else:
                self.stats_list = [self.stats_class(archive_tensors=self.archive_tensors)]
            self.initiated = True
        if self.tupled_input:
            for i in range(len(input)):
                self.stats_list[i].forward(input[i])
        else:
            self.stats_list[0].forward(input)

    def get_tensor_quant_params(self, ten_qconfig):
        ret_list = [stats_obj.get_tensor_quant_params(ten_qconfig) for stats_obj in self.stats_list]
        if self.tupled_input:
            return ret_list
        else:
            return ret_list[0]

    def print(self, name=''):
        for i in range(len(self.stats_list)):
            self.stats_list[i].print(name + '[{}]'.format(i))


class TensorChannelIntQuantParams(object):

    def __init__(self, min_vals, max_vals, qconfig):
        super(TensorChannelIntQuantParams, self).__init__()
        assert len(min_vals) == len(max_vals)
        self.num_channels = len(min_vals)
        self.channel_qparams = []
        for min_val, max_val in zip(min_vals, max_vals):
            self.channel_qparams.append(TensorFullIntQuantParams(min_val, max_val, qconfig))

    def quant_dequant(self, tensor_f):
        tensor_q = torch.zeros_like(tensor_f)
        for chan_id in range(self.num_channels):
            tensor_q[chan_id] = self.channel_qparams[chan_id].quant_dequant(tensor[chan_id])
        return tensor_q


class ChannleWiseMinMaxStats(torch.nn.Module):

    def __init__(self):
        super(ChannleWiseMinMaxStats, self).__init__()
        self.min_vals = None
        self.max_vals = None

    def forward(self, tensor):
        num_chans = tensor.shape[0]
        if self.min_vals == None or self.max_vals == None:
            self.min_vals = [None] * num_chans
            self.max_vals = [None] * num_chans
        for chan_id in range(num_chans):
            min_val = torch.min(tensor[chan_id]).item()
            max_val = torch.max(tensor[chan_id]).item()
            if self.min_vals[chan_id] == None:
                self.min_vals[chan_id] = min_val
            elif self.min_vals[chan_id] < min_val:
                self.min_vals[chan_id] = min_val
            if self.max_vals[chan_id] == None:
                self.max_vals[chan_id] = max_val
            elif self.max_vals[chan_id] > max_val:
                self.max_vals[chan_id] = max_val

    def get_tensor_quant_params(self, ten_qconfig):
        ten_qparams = TensorChannelIntQuantParams(self.min_vals, self.max_vals, ten_qconfig)
        return ten_qparams

    def print(self):
        None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddMatmul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (BatchMatmul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (BertAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, position_embedding_type=4, is_decoder=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOnlyNSPHead,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, position_embedding_type=4, is_decoder=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannleWiseMinMaxStats,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EltwiseDiv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Matmul,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mean,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MinMaxStats,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RunningMinMaxStats,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaleShift,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SparseLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TensorDumpListWrapper,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_IntelLabs_FP8_Emulation_Toolkit(_paritybench_base):
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

