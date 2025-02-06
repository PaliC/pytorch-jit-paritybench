import sys
_module = sys.modules[__name__]
del sys
msp = _module
pa100k = _module
peta = _module
petazs = _module
process_msp = _module
process_pa100k = _module
process_peta = _module
process_rap1 = _module
process_rap2 = _module
process_rapzs = _module
rap1 = _module
rap2 = _module
rapzs = _module
template = _module
config = _module
AttrDataset = _module
dataset = _module
infer = _module
local = _module
CE_loss = _module
loss = _module
make_optimizer = _module
common = _module
registry = _module
utils = _module
Qformer = _module
models = _module
base_model = _module
blip2 = _module
blip2_outputs = _module
eva_vit = _module
attr2vec_block = _module
cbam = _module
lora_layers = _module
tools = _module
function = _module
utils = _module
train = _module
loading = _module
logging = _module
train_utils = _module
vis_utils = _module
batch_engine = _module
batch_engine_KD = _module
AttrDataset = _module
preprocess = _module
pa100k = _module
peta = _module
rap = _module
wider = _module
CE_loss = _module
Vim = _module
base_block = _module
components = _module
csm_triton = _module
hybrid_1 = _module
hybrid_2 = _module
hybrid_3 = _module
hybrid_4 = _module
hybrid_5 = _module
hybrid_6 = _module
hybrid_7 = _module
hybrid_8 = _module
rope = _module
vit = _module
vmamba = _module
vmamba_checks = _module
solver = _module
cosine_lr = _module
lr_scheduler = _module
make_optimizer = _module
scheduler = _module
scheduler_factory = _module
function = _module
utils = _module
train = _module
train_hybrid = _module
batch_engine = _module
clip = _module
clip = _module
model = _module
simple_tokenizer = _module
AttrDataset = _module
pa100k_pad = _module
peta_pad = _module
petazspad = _module
rap1_pad = _module
rap2_pad = _module
rapzspad = _module
read = _module
upar = _module
wider_pad = _module
log_untils = _module
CE_loss = _module
base_block = _module
vit = _module
cosine_lr = _module
lr_scheduler = _module
make_optimizer = _module
scheduler = _module
test_example = _module
function = _module
utils = _module
train = _module
distributed = _module
file_io = _module
io_utils = _module
train_utils = _module
vis_utils = _module
batch_engine = _module
AttrDataset = _module
peta = _module
CE_loss = _module
SNN_model = _module
base_block = _module
vit = _module
cosine_lr = _module
lr_scheduler = _module
make_optimizer = _module
scheduler = _module
function = _module
utils = _module
train = _module
CLIP = _module
clip = _module
model = _module
batch_engine = _module
AttrDataset = _module
format_peta = _module
decoder = _module
attention = _module
decoders = _module
utils = _module
eval = _module
eval_batch = _module
log_utils = _module
CE_loss = _module
NLL_loss = _module
base_block = _module
vit = _module
cosine_lr = _module
lr_scheduler = _module
make_optimizer = _module
scheduler = _module
function = _module
utils = _module
train = _module
utils = _module
distributed = _module
train_utils = _module
typing = _module
vis_utils = _module
clip = _module
model = _module
batch_engine = _module
AttrDataset = _module
duke = _module
mars = _module
gui_detection = _module
test = _module
CE_loss = _module
CrossFrameSidenet = _module
attn_helper = _module
base_block = _module
layers = _module
sidenet = _module
sidenet_vit = _module
spatial_sidenet = _module
temporal_sidenet = _module
test_base = _module
timm_wrapper = _module
visual = _module
vit = _module
cosine_lr = _module
lr_scheduler = _module
make_optimizer = _module
scheduler = _module
function = _module
utils = _module
train = _module
distributed = _module
train_utils = _module
vis_utils = _module

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


import torch.utils.data as data


import torchvision.transforms as T


import random


from collections import OrderedDict


import torch


from torch.utils.data import DataLoader


import time


from torch.cuda.amp import autocast


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import logging


import re


from typing import Optional


import pandas as pd


from torch.utils.model_zoo import tqdm


from torchvision.datasets.utils import check_integrity


from torchvision.datasets.utils import download_file_from_google_drive


from torchvision.datasets.utils import extract_archive


import math


import warnings


from typing import Tuple


from typing import Dict


from typing import Any


from torch import Tensor


from torch import device


from torch import dtype


from torch import nn


import torch.utils.checkpoint


from torch.nn import CrossEntropyLoss


import torch.distributed as dist


from functools import partial


import torch.utils.checkpoint as checkpoint


import torch.nn.init as init


from typing import List


from torch.autograd import Variable


from collections import defaultdict


from collections import deque


from sklearn.metrics import confusion_matrix


from torch.nn.utils import clip_grad_norm_


from scipy.io import loadmat


from collections import namedtuple


import copy


from torch.nn.functional import mse_loss


from math import pi


from itertools import repeat


import collections.abc as container_abcs


from typing import Callable


from typing import Union


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from functools import reduce


from torch.nn import Conv2d


from torch.nn import Dropout


from itertools import count


from re import L


import numpy


from torch import optim


from torch.nn.parameter import Parameter


from torch.nn import NLLLoss


from torch.nn import functional as F


from collections import Counter


from typing import Sequence


from random import sample


from torchvision import transforms


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)
    weights[targets > 1] = 0.0
    return weights


class CEL_Sigmoid(nn.Module):

    def __init__(self, sample_weight=None, size_average=True, attr_idx=None):
        super(CEL_Sigmoid, self).__init__()
        self.sample_weight = sample_weight
        self.size_average = size_average
        self.attr_idx = attr_idx

    def forward(self, logits, targets):
        batch_size = logits.shape[0]
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            if self.attr_idx is not None and targets_mask.shape[1] != self.sample_weight.shape[0]:
                weight = ratio2weight(targets_mask[:, self.attr_idx], self.sample_weight)
                loss = loss[:, self.attr_idx]
            else:
                weight = ratio2weight(targets_mask, self.sample_weight)
            loss = loss * weight
        loss = loss.sum() / batch_size if self.size_average else loss.sum()
        return loss


class FocalLoss(nn.Module):

    def __init__(self, sample_weight=None, size_average=True, attr_idx=None, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.sample_weight = sample_weight
        self.size_average = size_average
        self.attr_idx = attr_idx
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        batch_size = logits.shape[0]
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        if self.sample_weight is not None:
            if self.attr_idx is not None and targets_mask.shape[1] != self.sample_weight.shape[0]:
                weight = ratio2weight(targets_mask[:, self.attr_idx], self.sample_weight)
                focal_loss = focal_loss[:, self.attr_idx]
            else:
                weight = ratio2weight(targets_mask, self.sample_weight)
            focal_loss = focal_loss * weight
        loss = focal_loss.sum() / batch_size if self.size_average else focal_loss.sum()
        return loss


class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.config = config

    def forward(self, input_ids=None, position_ids=None, query_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length].clone()
        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == 'absolute':
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings
            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
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
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = key_layer, value_layer
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
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
            attention_mask = attention_mask
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)
        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
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

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        if self.config.add_cross_attention and layer_num % self.config.cross_attention_freq == 0:
            self.crossattention = BertAttention(config, is_cross_attention=self.config.add_cross_attention)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, query_length=0):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]
            if self.has_cross_attention:
                assert encoder_hidden_states is not None, 'encoder_hidden_states must be given for cross-attention layers'
                cross_attention_outputs = self.crossattention(query_attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions)
                query_attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]
            layer_output = apply_chunking_to_forward(self.feed_forward_chunk_query, self.chunk_size_feed_forward, self.seq_len_dim, query_attention_output)
            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output[:, query_length:, :])
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True, query_length=0):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                if use_cache:
                    logger.warn('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, query_length)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += layer_outputs[-1],
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
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

