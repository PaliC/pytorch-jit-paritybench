import sys
_module = sys.modules[__name__]
del sys
pytorch_volumetric = _module
chamfer = _module
model_to_sdf = _module
sdf = _module
visualization = _module
volume = _module
voxel = _module
test_export_composed_sdf = _module
test_chamfer = _module
test_model_to_sdf = _module
test_sdf = _module
test_voxel_sdf = _module

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


import time


from typing import NamedTuple


import torch


import typing


import numpy as np


import logging


import abc


import enum


import math


from typing import Union


from functools import partial


import copy


from matplotlib import pyplot as plt


import matplotlib.colors


import matplotlib.pyplot as plt


import matplotlib


from matplotlib import cm

