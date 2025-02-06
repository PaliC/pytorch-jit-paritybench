import sys
_module = sys.modules[__name__]
del sys
onnx_model_maker = _module
code_gen = _module
ops = _module
op_helper = _module
op_ver_1 = _module
op_ver_10 = _module
op_ver_11 = _module
op_ver_12 = _module
op_ver_13 = _module
op_ver_14 = _module
op_ver_15 = _module
op_ver_16 = _module
op_ver_17 = _module
op_ver_2 = _module
op_ver_3 = _module
op_ver_4 = _module
op_ver_5 = _module
op_ver_6 = _module
op_ver_7 = _module
op_ver_8 = _module
op_ver_9 = _module
onnx_pytorch = _module
_version = _module
code_gen = _module
code_gen_template = _module
Abs = _module
Acos = _module
Acosh = _module
Add = _module
And = _module
ArgMax = _module
ArgMin = _module
Asin = _module
Asinh = _module
Atan = _module
Atanh = _module
AveragePool = _module
BatchNormalization = _module
BitShift = _module
Cast = _module
Ceil = _module
Clip = _module
Concat = _module
Constant = _module
ConstantOfShape = _module
Conv = _module
ConvTranspose = _module
Cos = _module
Cosh = _module
Div = _module
Dropout = _module
Elu = _module
Equal = _module
Exp = _module
Expand = _module
Flatten = _module
Floor = _module
Gather = _module
GatherND = _module
Gemm = _module
GlobalAveragePool = _module
Greater = _module
Identity = _module
InstanceNormalization = _module
LRN = _module
LayerNormalization = _module
LeakyRelu = _module
Less = _module
Log = _module
MatMul = _module
Max = _module
MaxPool = _module
Mul = _module
NonMaxSuppression = _module
NonZero = _module
Not = _module
PRelu = _module
Pad = _module
Reciprocal = _module
ReduceMean = _module
ReduceMin = _module
ReduceProd = _module
ReduceSum = _module
Relu = _module
Reshape = _module
Resize = _module
RoiAlign = _module
Round = _module
Scatter = _module
ScatterElements = _module
Shape = _module
Sigmoid = _module
Slice = _module
Softmax = _module
Split = _module
Sqrt = _module
Squeeze = _module
Sub = _module
Tanh = _module
TopK = _module
Transpose = _module
Unsqueeze = _module
Upsample = _module
op_code_generators = _module
tests = _module
test_base = _module
test_onnx_model_zoo = _module
utils = _module
embedding_config_helper = _module
setup = _module
tutorial = _module

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


import re


from collections import Counter


import numpy as np


import torch

