import sys
_module = sys.modules[__name__]
del sys
deepdoctection = _module
analyzer = _module
_config = _module
dd = _module
factory = _module
configs = _module
dataflow = _module
base = _module
common = _module
custom = _module
custom_serialize = _module
parallel_map = _module
serialize = _module
stats = _module
datapoint = _module
annotation = _module
box = _module
convert = _module
image = _module
view = _module
datasets = _module
adapter = _module
dataflow_builder = _module
info = _module
instances = _module
doclaynet = _module
fintabnet = _module
funsd = _module
iiitar13k = _module
layouttest = _module
publaynet = _module
pubtables1m = _module
pubtabnet = _module
rvlcdip = _module
xfund = _module
xsl = _module
registry = _module
save = _module
eval = _module
accmetric = _module
cocometric = _module
tedsmetric = _module
tp_eval_callback = _module
extern = _module
base = _module
d2detect = _module
deskew = _module
doctrocr = _module
fastlang = _module
hfdetr = _module
hflayoutlm = _module
hflm = _module
model = _module
pdftext = _module
pt = _module
nms = _module
ptutils = _module
tessocr = _module
texocr = _module
tp = _module
tfutils = _module
tpcompat = _module
tpfrcnn = _module
config = _module
modeling = _module
backbone = _module
generalized_rcnn = _module
model_box = _module
model_cascade = _module
model_fpn = _module
model_frcnn = _module
model_mrcnn = _module
model_rpn = _module
predict = _module
preproc = _module
utils = _module
box_ops = _module
np_box_ops = _module
tpdetect = _module
mapper = _module
cats = _module
cocostruct = _module
d2struct = _module
hfstruct = _module
laylmstruct = _module
maputils = _module
match = _module
misc = _module
pascalstruct = _module
prodigystruct = _module
pubstruct = _module
tpstruct = _module
xfundstruct = _module
pipe = _module
anngen = _module
concurrency = _module
doctectionpipe = _module
language = _module
layout = _module
lm = _module
order = _module
refine = _module
segment = _module
sub_layout = _module
text = _module
transform = _module
train = _module
d2_frcnn_train = _module
hf_detr_train = _module
hf_layoutlm_train = _module
tp_frcnn_train = _module
context = _module
develop = _module
env_info = _module
error = _module
file_utils = _module
fs = _module
identifier = _module
logger = _module
metacfg = _module
mocks = _module
pdf_utils = _module
settings = _module
tqdm = _module
types = _module
viz = _module
export_tracing_d2 = _module
reduce_d2 = _module
reduce_tp = _module
tp2d2 = _module
setup = _module
tests = _module
test_dd = _module
conftest = _module
data = _module
test_common = _module
test_custom = _module
test_custom_serialize = _module
test_parallel_map = _module
test_stats = _module
test_annotation = _module
test_box = _module
test_convert = _module
test_image = _module
test_view = _module
test_doclaynet = _module
test_fintabnet = _module
test_funsd = _module
test_iiitar13k = _module
test_layouttest = _module
test_publaynet = _module
test_pubtables1m = _module
test_pubtabnet = _module
test_rvlcdip = _module
test_adapter = _module
test_info = _module
test_registry = _module
test_accmetric = _module
test_cocometric = _module
test_eval = _module
test_tedsmetric = _module
test_base = _module
test_deskew = _module
test_doctrocr = _module
test_fastlang = _module
test_hfdetr = _module
test_hflayoutlm = _module
test_hflm = _module
test_model = _module
test_pdftext = _module
test_tessocr = _module
test_texocr = _module
test_tpdetect = _module
test_cats = _module
test_cocostruct = _module
test_d2struct = _module
test_hfstruct = _module
test_laylmstruct = _module
test_match = _module
test_misc = _module
test_prodigystruct = _module
test_pubstruct = _module
test_tpstruct = _module
test_utils = _module
test_xfundstruct = _module
test_anngen = _module
test_doctectionpipe = _module
test_language = _module
test_layout = _module
test_lm = _module
test_order = _module
test_refine = _module
test_segment = _module
test_sub_layout = _module
test_text = _module
test_transform = _module
test_d2_frcnn_train = _module
test_tp_frcnn_train = _module
tests_d2 = _module
test_d2detect = _module

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


from typing import Any


from typing import Callable


from typing import Iterator


from typing import Mapping


from typing import Optional


from typing import Union


from abc import ABC


from abc import abstractmethod


from types import MappingProxyType


from typing import TYPE_CHECKING


from typing import Literal


from typing import Sequence


from typing import overload


from copy import copy


import numpy as np


from collections import defaultdict


import random


from typing import NewType


import numpy.typing as npt


import copy


from typing import Type


import re


import string


from types import ModuleType


from typing import no_type_check


import torch


from typing import Dict


from typing import List

