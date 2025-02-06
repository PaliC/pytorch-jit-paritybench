import sys
_module = sys.modules[__name__]
del sys
config_mountaincar = _module
run_mountain_car_multiple = _module
run_mountaincar = _module
config_pendulum = _module
run_pendulum = _module
run_pendulum_multiple = _module
config_process_control = _module
run_process_control = _module
run_processc_control_multiple = _module
rl_gp_mpc = _module
actions_config = _module
controller_config = _module
memory_config = _module
model_config = _module
observation_config = _module
reward_config = _module
total_config = _module
training_config = _module
functions_process_config = _module
visu_config = _module
control_objects = _module
abstract_action_mapper = _module
action_init_functions = _module
derivative_action_mapper = _module
normalization_action_mapper = _module
abstract_controller = _module
gp_mpc_controller = _module
iteration_info_class = _module
gp_memory = _module
abstract_model = _module
gp_model = _module
abstract_observation_state_mapper = _module
normalization_observation_state_mapper = _module
abstract_state_reward_mapper = _module
setpoint_distance_reward_mapper = _module
data_utils = _module
pytorch_utils = _module
process_control = _module
run_env_function = _module
dynamic_2d_graph = _module
static_2d_graph = _module
static_3d_graph = _module
utils = _module
visu_object = _module

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


from typing import Union


import numpy as np


from scipy.optimize import minimize


import time


from math import sqrt


import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline


from sklearn.preprocessing import StandardScaler


from sklearn.neighbors import KNeighborsRegressor

