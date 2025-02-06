import sys
_module = sys.modules[__name__]
del sys
conf = _module
enjoy = _module
ppo_config_example = _module
rl_zoo3 = _module
benchmark = _module
callbacks = _module
cli = _module
enjoy = _module
exp_manager = _module
gym_patches = _module
hyperparams_opt = _module
import_envs = _module
load_from_hub = _module
plots = _module
all_plots = _module
plot_from_file = _module
plot_train = _module
score_normalization = _module
push_to_hub = _module
record_training = _module
record_video = _module
train = _module
utils = _module
wrappers = _module
scripts = _module
create_cluster_jobs = _module
create_mujoco_jobs = _module
migrate_to_hub = _module
parse_study = _module
run_jobs = _module
setup = _module
test_env = _module
config = _module
test_callbacks = _module
test_enjoy = _module
test_hyperparams_opt = _module
test_train = _module
test_wrappers = _module

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


import torch as th


import time


import warnings


from collections import OrderedDict


from typing import Any


from typing import Callable


from typing import Optional


from typing import Union


from torch import nn as nn


from copy import deepcopy


import uuid

