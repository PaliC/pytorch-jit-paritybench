import sys
_module = sys.modules[__name__]
del sys
conf = _module
tests = _module
test_bicop = _module
test_vinecop = _module
torchvinecopulib = _module
bicop = _module
_abc = _module
_archimedean = _module
_clayton = _module
_data_bcp = _module
_elliptical = _module
_factory_bcp = _module
_frank = _module
_gaussian = _module
_gumbel = _module
_independent = _module
_joe = _module
_studentt = _module
util = _module
vinecop = _module
_data_vcp = _module
_factory_vcp = _module

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


import logging


import matplotlib.pyplot as plt


import numpy as np


import pandas as pd


import torch


import random


from itertools import combinations


from scipy.stats import kendalltau


from abc import ABC


from abc import abstractmethod


from math import log1p


import math


from enum import Enum


from scipy.optimize import minimize


from math import exp


from math import expm1


from math import sqrt


from math import ceil


from math import floor


from math import lgamma


from math import log


from functools import partial


from itertools import product


from scipy.special import stdtr


from scipy.special import stdtrit


from collections import defaultdict


from collections import deque


from random import seed as r_seed


from typing import Deque

