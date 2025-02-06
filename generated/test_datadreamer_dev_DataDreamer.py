import sys
_module = sys.modules[__name__]
del sys
source = _module
conf = _module
docstrings = _module
__cli__ = _module
src = _module
_cachable = _module
_cachable = _module
_parallel_cachable = _module
_patches = _module
datasets_reset_state_hack = _module
setfit_import_hack = _module
datadreamer = _module
datasets = _module
datasets = _module
utils = _module
embedders = _module
embedder = _module
openai_embedder = _module
parallel_embedder = _module
sentence_transformers_embedder = _module
together_embedder = _module
errors = _module
steps = _module
step = _module
llms = _module
_chat_prompt_templates = _module
_litellm = _module
_llm_api = _module
_tokenizers = _module
ai21 = _module
anthropic = _module
bedrock = _module
cohere = _module
ctransformers = _module
google_ai_studio = _module
hf_api_endpoint = _module
hf_transformers = _module
llm = _module
mistral_ai = _module
openai = _module
openai_assistant = _module
parallel_llm = _module
petals = _module
together = _module
vertex_ai = _module
vllm = _module
logging = _module
logger = _module
pickling = _module
pickle = _module
project = _module
builtin_tasks = _module
debug = _module
devices = _module
environment = _module
pennnlp = _module
persistent_storage = _module
report = _module
serve = _module
retrievers = _module
embedding_retriever = _module
parallel_retriever = _module
retriever = _module
data_card = _module
csv_data_source = _module
data_source = _module
hf_dataset_data_source = _module
hf_hub_data_source = _module
json_data_source = _module
text_data_source = _module
_prompt_base = _module
data_from_attributed_prompt = _module
data_from_prompt = _module
few_shot_prompt = _module
few_shot_prompt_with_retrieval = _module
filter_with_prompt = _module
judge_generation_pairs_with_prompt = _module
judge_pairs_with_prompt = _module
process_with_prompt = _module
prompt = _module
rag_prompt = _module
rank_with_prompt = _module
step = _module
step_background = _module
step_export = _module
step_operations = _module
step_output = _module
cosine_similarity = _module
embed = _module
retrieve = _module
run_task_model = _module
task_models = _module
hf_classification_task_model = _module
parallel_task_model = _module
task_model = _module
tests = _module
conftest = _module
test_datasets = _module
test_utils = _module
test_embedders = _module
test_llms = _module
test_retrievers = _module
test_prompt = _module
tasks = _module
test_tasks = _module
test_data_sources = _module
test_step = _module
test_step_background = _module
test_step_export = _module
test_step_operations = _module
test_step_output = _module
test_task_models = _module
test_cli = _module
test_datadreamer = _module
test_package = _module
config = _module
fixtures = _module
bitsandbytes_fixture = _module
clear_space = _module
cli_runner = _module
create_datadreamer = _module
create_test_step = _module
mock_llm = _module
restore_os_environ = _module
trainers = _module
test_distributed = _module
test_trainers = _module
test_device_utils = _module
_train_hf_base = _module
_vendored = _module
_dpo_helper = _module
_sentence_transformer_helper = _module
_setfit_helper = _module
dpo_trainer = _module
train_hf_classifier = _module
train_hf_dpo = _module
train_hf_finetune = _module
train_hf_ppo = _module
train_hf_reward_model = _module
train_openai_finetune = _module
train_sentence_transformer = _module
train_setfit_classifier = _module
trainer = _module
arg_utils = _module
background_utils = _module
collection_utils = _module
device_utils = _module
distributed_utils = _module
fingerprint_utils = _module
fs_utils = _module
hf_chat_prompt_templates = _module
hf_hub_utils = _module
hf_model_utils = _module
hf_structured_decoding_utils = _module
hf_training_utils = _module
import_utils = _module
ring_utils = _module
str_utils = _module
time_utils = _module

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


import inspect


import itertools


from functools import cached_property


from functools import partial


import logging


from abc import ABC


from abc import abstractmethod


from collections import Counter


from collections import defaultdict


from collections.abc import Iterator


from collections.abc import Sized


from itertools import chain


from itertools import islice


from itertools import tee


from logging import Logger


from math import ceil


from time import time


from typing import Any


from typing import Callable


from typing import DefaultDict


from typing import Generator


from typing import Iterable


from typing import cast


from uuid import uuid4


import torch


from collections import UserDict


from typing import TYPE_CHECKING


from collections.abc import Iterable


from pandas import DataFrame


import numpy as np


import torch._dynamo


from functools import lru_cache


import re


from types import MethodType


from functools import cache


from logging import root


import warnings


from typing import Sequence


import torch.nn.functional as F


import typing


import uuid


from math import floor


from random import Random


from time import sleep


from types import GeneratorType


import time


from types import SimpleNamespace


from copy import copy


from typing import Type


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


from torch.nn.utils.rnn import pad_sequence


import random


from copy import deepcopy


from functools import wraps


from typing import Literal


from typing import Tuple


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.nn import functional as F


from torch.optim import AdamW


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LRScheduler


from functools import total_ordering


from collections import namedtuple


from logging import StreamHandler


import torch.cuda


import math


from torch.optim.lr_scheduler import LambdaLR


from types import ModuleType


class SentenceTransformerLossWrapper(torch.nn.Module):

    def __init__(self, orig_model: 'SentenceTransformer', wrapped_model: 'SentenceTransformerWrapper', loss_module: 'torch.nn.Module', _is_peft: 'bool'):
        torch.nn.Module.__init__(self)
        self.orig_model = orig_model
        self.wrapped_model = wrapped_model
        self.loss_module = loss_module
        self._is_peft = _is_peft

    def __getattr__(self, name):
        if name == 'config':
            if self._is_peft:
                sentence_transformer_model = get_base_model_from_peft_model(self.orig_model)
            else:
                sentence_transformer_model = self.orig_model
            has_transformer_module = '0' in sentence_transformer_model._modules and isinstance(sentence_transformer_model._modules['0'], Transformer)
            if has_transformer_module:
                transformer_module = sentence_transformer_model._modules['0']
                has_auto_model = 'auto_model' in transformer_module._modules and isinstance(transformer_module._modules['auto_model'], PreTrainedModel)
                if has_auto_model:
                    return transformer_module._modules['auto_model'].config
        return super().__getattr__(name)

    def forward(self, anchor_input_ids: 'None | torch.Tensor'=None, anchor_attention_mask: 'None | torch.Tensor'=None, positive_input_ids: 'None | torch.Tensor'=None, positive_attention_mask: 'None | torch.Tensor'=None, negative_input_ids: 'None | torch.Tensor'=None, negative_attention_mask: 'None | torch.Tensor'=None, labels: 'None | torch.Tensor'=None, num_items_in_batch=None):
        _uniq_ids = []
        sentence_features = []
        _uniq_ids.append(uuid4().hex)
        sentence_features.append({'_uniq_id': _uniq_ids[-1], 'input_ids': anchor_input_ids, 'attention_mask': anchor_attention_mask})
        if positive_input_ids is not None:
            _uniq_ids.append(uuid4().hex)
            sentence_features.append({'_uniq_id': _uniq_ids[-1], 'input_ids': positive_input_ids, 'attention_mask': positive_attention_mask})
        if negative_input_ids is not None:
            _uniq_ids.append(uuid4().hex)
            sentence_features.append({'_uniq_id': _uniq_ids[-1], 'input_ids': negative_input_ids, 'attention_mask': negative_attention_mask})
        loss = self.loss_module(sentence_features=sentence_features, labels=labels)
        return {'loss': loss, 'embeddings': [self.wrapped_model.results[_uniq_id]['sentence_embedding'].detach() for _uniq_id in _uniq_ids], 'loss_for_joint_metric': loss}

