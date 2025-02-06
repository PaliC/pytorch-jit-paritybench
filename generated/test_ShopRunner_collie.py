import sys
_module = sys.modules[__name__]
del sys
collie = _module
_version = _module
config = _module
cross_validation = _module
interactions = _module
dataloaders = _module
datasets = _module
samplers = _module
loss = _module
bpr = _module
hinge = _module
metadata_utils = _module
warp = _module
metrics = _module
model = _module
base = _module
base_pipeline = _module
layers = _module
multi_stage_pipeline = _module
trainer = _module
cold_start_matrix_factorization = _module
collaborative_metric_learning = _module
deep_fm = _module
hybrid_matrix_factorization = _module
hybrid_pretrained_matrix_factorization = _module
matrix_factorization = _module
mlp_matrix_factorization = _module
neural_collaborative_filtering = _module
nonlinear_matrix_factorization = _module
movielens = _module
get_data = _module
run = _module
visualize = _module
utils = _module
conf = _module
setup = _module
tests = _module
conftest = _module
fixtures = _module
cross_validation_fixtures = _module
interactions_fixtures = _module
loss_fixtures = _module
metrics_fixtures = _module
model_fixtures = _module
movielens_fixtures = _module
utils_fixtures = _module
test_cross_validation = _module
test_docstring = _module
test_interactions = _module
test_losses = _module
test_metrics = _module
test_model = _module
test_movielens = _module
test_utils = _module

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


from typing import Iterable


from typing import Optional


from typing import Union


import numpy as np


from scipy.sparse import coo_matrix


import torch


from abc import ABCMeta


from abc import abstractmethod


import collections


import random


from typing import Any


from typing import List


from typing import Tuple


import warnings


import pandas as pd


from scipy.sparse import dok_matrix


import math


from typing import Dict


from typing import Callable


from scipy.sparse import csr_matrix


from collections.abc import Iterable


from collections import OrderedDict


from functools import reduce


from functools import partial


from torch import nn


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.nn.functional as F


import copy


import inspect


import re


import time


from numpy.testing import assert_almost_equal


from numpy.testing import assert_array_equal


from sklearn.metrics import roc_auc_score


from torch.optim.lr_scheduler import StepLR


class ScaledEmbedding(torch.nn.Embedding):
    """Embedding layer that initializes its values to use a truncated normal distribution."""

    def reset_parameters(self) ->None:
        """Overriding default ``reset_parameters`` method."""
        self.weight.data.normal_(0, 1.0 / (self.embedding_dim * 2.5))


class ZeroEmbedding(torch.nn.Embedding):
    """Embedding layer with weights zeroed-out."""

    def reset_parameters(self) ->None:
        """Overriding default ``reset_parameters`` method."""
        self.weight.data.zero_()

