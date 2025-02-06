import sys
_module = sys.modules[__name__]
del sys
cv2_util = _module
coco_eval = _module
coco_utils = _module
engine = _module
group_by_aspect_ratio = _module
train = _module
transforms = _module
utils = _module
farst_predict = _module
predict = _module
coco_eval = _module
coco_utils = _module
engine = _module
group_by_aspect_ratio = _module
train = _module
transforms = _module
utils = _module
predict = _module
copy = _module
draw = _module
new_json_to_dataset = _module

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


import copy


import time


import torch


from collections import defaultdict


import torch.utils.data


import torchvision


import math


import torchvision.models.detection.mask_rcnn


from itertools import repeat


from itertools import chain


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import Sampler


from torch.utils.model_zoo import tqdm


from torch import nn


import torchvision.models.detection


import random


from torchvision.transforms import functional as F


from collections import deque


import torch.distributed as dist


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

