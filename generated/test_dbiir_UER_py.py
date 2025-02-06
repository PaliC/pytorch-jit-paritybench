import sys
_module = sys.modules[__name__]
del sys
run_c3 = _module
run_chid = _module
run_classifier = _module
run_classifier_cv = _module
run_classifier_grid = _module
run_classifier_mt = _module
run_classifier_multi_label = _module
run_classifier_prompt = _module
run_classifier_siamese = _module
run_cmrc = _module
run_dbqa = _module
run_ner = _module
run_regression = _module
run_simcse = _module
run_text2text = _module
run_c3_infer = _module
run_chid_infer = _module
run_classifier_cv_infer = _module
run_classifier_infer = _module
run_classifier_mt_infer = _module
run_classifier_multi_label_infer = _module
run_classifier_prompt_infer = _module
run_classifier_siamese_infer = _module
run_cmrc_infer = _module
run_ner_infer = _module
run_regression_infer = _module
run_text2text_infer = _module
preprocess = _module
pretrain = _module
scripts = _module
average_models = _module
build_vocab = _module
cloze_test = _module
convert_albert_from_huggingface_to_uer = _module
convert_albert_from_original_tf_to_uer = _module
convert_albert_from_uer_to_huggingface = _module
convert_albert_from_uer_to_original_tf = _module
convert_bart_from_huggingface_to_uer = _module
convert_bart_from_uer_to_huggingface = _module
convert_bert_extractive_qa_from_huggingface_to_uer = _module
convert_bert_extractive_qa_from_uer_to_huggingface = _module
convert_bert_from_huggingface_to_uer = _module
convert_bert_from_original_tf_to_uer = _module
convert_bert_from_uer_to_huggingface = _module
convert_bert_from_uer_to_original_tf = _module
convert_bert_text_classification_from_huggingface_to_uer = _module
convert_bert_text_classification_from_uer_to_huggingface = _module
convert_bert_token_classification_from_huggingface_to_uer = _module
convert_bert_token_classification_from_uer_to_huggingface = _module
convert_gpt2_from_huggingface_to_uer = _module
convert_gpt2_from_uer_to_huggingface = _module
convert_pegasus_from_huggingface_to_uer = _module
convert_pegasus_from_uer_to_huggingface = _module
convert_sbert_from_huggingface_to_uer = _module
convert_sbert_from_uer_to_huggingface = _module
convert_t5_from_huggingface_to_uer = _module
convert_t5_from_uer_to_huggingface = _module
convert_xlmroberta_from_huggingface_to_uer = _module
convert_xlmroberta_from_uer_to_huggingface = _module
diff_vocab = _module
dynamic_vocab_adapter = _module
extract_embeddings = _module
extract_features = _module
generate_lm = _module
generate_seq2seq = _module
run_lgb = _module
run_lgb_cv_bayesopt = _module
topn_words_dep = _module
topn_words_indep = _module
uer = _module
decoders = _module
transformer_decoder = _module
embeddings = _module
dual_embedding = _module
embedding = _module
pos_embedding = _module
seg_embedding = _module
sinusoidalpos_embedding = _module
word_embedding = _module
encoders = _module
cnn_encoder = _module
dual_encoder = _module
rnn_encoder = _module
transformer_encoder = _module
initialize = _module
layers = _module
layer_norm = _module
multi_headed_attn = _module
position_ffn = _module
relative_position_embedding = _module
transformer = _module
model_builder = _module
model_loader = _module
model_saver = _module
models = _module
model = _module
opts = _module
targets = _module
bilm_target = _module
cls_target = _module
lm_target = _module
mlm_target = _module
sp_target = _module
target = _module
trainer = _module
utils = _module
act_fun = _module
adversarial = _module
config = _module
constants = _module
dataloader = _module
dataset = _module
logging = _module
mask = _module
misc = _module
optimizers = _module
seed = _module
tokenizers = _module
vocab = _module

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


import random


import torch


import torch.nn as nn


import numpy as np


from itertools import product


import time


import re


import logging


import collections


import torch.nn.functional as F


from scipy.stats import spearmanr


import math


import scipy.stats


import tensorflow as tf


from tensorflow.python import pywrap_tensorflow


import tensorflow.keras.backend as K


import copy


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel


from typing import Callable


from typing import Iterable


from typing import Tuple


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    https://arxiv.org/abs/1607.06450
    """

    def __init__(self, hidden_size, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        hidden_states = self.gamma * (x - mean) / (std + self.eps)
        return hidden_states + self.beta


class Embedding(nn.Module):

    def __init__(self, args):
        super(Embedding, self).__init__()
        self.embedding_name_list = []
        self.dropout = nn.Dropout(args.dropout)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm and 'dual' not in args.embedding:
            self.layer_norm = LayerNorm(args.emb_size)

    def update(self, embedding, embedding_name):
        setattr(self, embedding_name, embedding)
        self.embedding_name_list.append(embedding_name)

    def forward(self, src, seg):
        if self.embedding_name_list[0] == 'dual':
            return self.dual(src, seg)
        for i, embedding_name in enumerate(self.embedding_name_list):
            embedding = getattr(self, embedding_name)
            if i == 0:
                emb = embedding(src, seg)
            else:
                emb += embedding(src, seg)
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb

