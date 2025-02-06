import sys
_module = sys.modules[__name__]
del sys
dtrocr = _module
config = _module
data = _module
model = _module
processor = _module
tests = _module
test_model = _module
test_processor = _module

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


from typing import Optional


from typing import Union


from typing import List


from torch import nn


from torch import Tensor


from typing import Tuple


from typing import Dict


from typing import Any


import time


import random


@dataclass
class DTrOCRModelOutput:
    hidden_states: 'torch.FloatTensor'
    past_key_values: 'torch.FloatTensor'


class DTrOCRModel(nn.Module):

    def __init__(self, config: 'DTrOCRConfig'):
        super().__init__()
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.hidden_layers = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self._attn_implementation = config._attn_implementation
        self.initialise_weights(config)

    def forward(self, pixel_values: 'torch.Tensor', input_ids: 'torch.LongTensor', position_ids: 'Optional[torch.LongTensor]'=None, past_key_values: 'Optional[Tuple[Tuple[torch.Tensor]]]'=None, attention_mask: 'Optional[torch.Tensor]'=None, use_cache: 'Optional[bool]'=False) ->DTrOCRModelOutput:
        device = input_ids.device if input_ids is not None else input_ids.device
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.hidden_layers))
        else:
            past_length = past_key_values[0][0].size(-2)
        patch_embeddings = self.patch_embeddings(pixel_values) if past_length == 0 else None
        token_embeddings = self.token_embedding(input_ids)
        if patch_embeddings is not None:
            patch_and_token_embeddings = torch.concat([patch_embeddings, token_embeddings], dim=-2)
        else:
            patch_and_token_embeddings = token_embeddings
        input_shape = patch_and_token_embeddings.shape
        if position_ids is None or past_length == 0:
            position_ids = torch.arange(past_length, input_shape[1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = torch.ones_like(position_ids, device=position_ids.device) * past_length
        position_embeddings = self.positional_embedding(position_ids)
        hidden_states = patch_and_token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)
        if attention_mask is not None:
            attention_mask = torch.concat([torch.ones(attention_mask.shape[0], patch_embeddings.shape[-2] if patch_embeddings is not None else past_length, dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=-1)
            if self._attn_implementation == 'flash_attention_2':
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask=attention_mask, input_shape=(input_shape[0], input_shape[-2]), inputs_embeds=patch_and_token_embeddings, past_key_values_length=past_length)
        presents = () if use_cache else None
        for hidden_layer, layer_past in zip(self.hidden_layers, past_key_values):
            outputs = hidden_layer(hidden_states, layer_past=layer_past, attention_mask=attention_mask, use_cache=use_cache)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        return DTrOCRModelOutput(hidden_states=hidden_states, past_key_values=presents)

    def initialise_weights(self, config: 'DTrOCRConfig') ->None:
        pretrained_gpt2 = GPT2Model.from_pretrained(config.gpt2_hf_model)
        for hidden_layer, pretrained_hidden_layer in zip(self.hidden_layers, pretrained_gpt2.h):
            hidden_layer.load_state_dict(pretrained_hidden_layer.state_dict())
        self.token_embedding.load_state_dict(pretrained_gpt2.wte.state_dict())

