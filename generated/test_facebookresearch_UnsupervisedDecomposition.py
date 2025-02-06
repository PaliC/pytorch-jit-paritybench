import sys
_module = sys.modules[__name__]
del sys
make_ref = _module
preprocess = _module
src = _module
data = _module
dataset = _module
dictionary = _module
loader = _module
evaluation = _module
evaluator = _module
glue = _module
xnli = _module
logger = _module
model = _module
embedder = _module
memory = _module
memory = _module
query = _module
utils = _module
pretrain = _module
transformer = _module
optim = _module
slurm = _module
trainer = _module
utils = _module
lowercase_and_remove_accent = _module
segment_th = _module
train = _module
translate = _module
transformers = _module
add_umt_subqs_subas_to_q_squad_format_new = _module
convert_hotpot2questionlist_script = _module
convert_hotpot2squad_simple_script = _module
conf = _module
download_glue_data = _module
ensemble_answers_by_confidence_script = _module
dataset = _module
distiller = _module
binarized_data = _module
extract_for_distil = _module
token_counts = _module
train = _module
utils = _module
hotpot_evaluate_v1 = _module
finetune_on_pregenerated = _module
pregenerate_training_data = _module
simple_lm_finetuning = _module
run_bertology = _module
run_generation = _module
run_glue = _module
run_lm_finetuning = _module
run_squad = _module
run_openai_gpt = _module
run_swag = _module
run_transfo_xl = _module
test_examples = _module
utils_glue = _module
utils_squad = _module
utils_squad_evaluate = _module
hubconf = _module
pseudoalignment = _module
embed_questions_with_bert = _module
pseudo_decomp_bert = _module
pseudo_decomp_bert_nsp = _module
pseudo_decomp_fasttext = _module
pseudo_decomp_random = _module
pseudo_decomp_tfidf = _module
pseudo_decomp_utils = _module
pseudo_decomp_variable = _module
replace_subq_entities = _module
pytorch_transformers = _module
convert_gpt2_checkpoint_to_pytorch = _module
convert_openai_checkpoint_to_pytorch = _module
convert_pytorch_checkpoint_to_tf = _module
convert_roberta_checkpoint_to_pytorch = _module
convert_tf_checkpoint_to_pytorch = _module
convert_transfo_xl_checkpoint_to_pytorch = _module
convert_xlm_checkpoint_to_pytorch = _module
convert_xlnet_checkpoint_to_pytorch = _module
file_utils = _module
modeling_auto = _module
modeling_bert = _module
modeling_distilbert = _module
modeling_gpt2 = _module
modeling_openai = _module
modeling_roberta = _module
modeling_transfo_xl = _module
modeling_transfo_xl_utilities = _module
modeling_utils = _module
modeling_xlm = _module
modeling_xlnet = _module
optimization = _module
tests = _module
conftest = _module
modeling_auto_test = _module
modeling_bert_test = _module
modeling_common_test = _module
modeling_distilbert_test = _module
modeling_gpt2_test = _module
modeling_openai_test = _module
modeling_roberta_test = _module
modeling_transfo_xl_test = _module
modeling_xlm_test = _module
modeling_xlnet_test = _module
optimization_test = _module
tokenization_auto_test = _module
tokenization_bert_test = _module
tokenization_dilbert_test = _module
tokenization_gpt2_test = _module
tokenization_openai_test = _module
tokenization_roberta_test = _module
tokenization_tests_commons = _module
tokenization_transfo_xl_test = _module
tokenization_utils_test = _module
tokenization_xlm_test = _module
tokenization_xlnet_test = _module
tokenization_auto = _module
tokenization_bert = _module
tokenization_distilbert = _module
tokenization_gpt2 = _module
tokenization_openai = _module
tokenization_roberta = _module
tokenization_transfo_xl = _module
tokenization_utils = _module
tokenization_xlm = _module
tokenization_xlnet = _module
setup = _module
split_hotpot_dev = _module
split_umt_dev = _module
umt_gen_subqs_to_squad_format = _module

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


from logging import getLogger


import math


import numpy as np


import torch


from collections import OrderedDict


import copy


import time


from torch import nn


import torch.nn.functional as F


from scipy.stats import spearmanr


from scipy.stats import pearsonr


from sklearn.metrics import f1_score


from sklearn.metrics import matthews_corrcoef


import itertools


from torch.nn import functional as F


import torch.nn as nn


import re


import inspect


from torch import optim


from torch.nn.utils import clip_grad_norm_


import random


from typing import List


from itertools import chain


from collections import Counter


import logging


from collections import namedtuple


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data import Subset


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from sklearn.preprocessing import normalize


import tensorflow as tf


import numpy


from functools import wraps


import collections


from torch.nn.parameter import Parameter


from collections import defaultdict


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


import uuid

