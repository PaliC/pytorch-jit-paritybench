import sys
_module = sys.modules[__name__]
del sys
check_loader_patches = _module
init = _module
check_loader_patches = _module
check_loader_patches = _module
check_resolution = _module
check_loader_patches = _module
networks = _module
organize_folder_structure = _module
predict_single_image = _module
train = _module
networks = _module
predict_single_image = _module
train = _module
predict_single_image = _module
train = _module
networks = _module
predict_single_image = _module
train = _module
utils = _module
networks = _module
predict_single_image = _module
train = _module
utils = _module

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


import matplotlib.pyplot as plt


import torch


from torch.utils.data import DataLoader


from torch.nn import init


from torch.optim import lr_scheduler


from torch.nn import Module


from torch.nn import Sequential


from torch.nn import Conv3d


from torch.nn import ConvTranspose3d


from torch.nn import BatchNorm3d


from torch.nn import MaxPool3d


from torch.nn import AvgPool1d


from torch.nn import Dropout3d


from torch.nn import ReLU


from torch.nn import Sigmoid


from collections import OrderedDict


import logging


from torch.utils.tensorboard import SummaryWriter


import re


import random


import time

