import sys
_module = sys.modules[__name__]
del sys
main = _module
data = _module
loss = _module
mixup = _module
model = _module

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


import torch.backends.cuda


import torch.backends.cudnn


from functools import partial


from typing import Optional


from typing import Sequence


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.datasets import CIFAR10


from torchvision.datasets import CIFAR100


from torchvision.datasets import DTD


from torchvision.datasets import STL10


from torchvision.datasets import FGVCAircraft


from torchvision.datasets import Flowers102


from torchvision.datasets import Food101


from torchvision.datasets import ImageFolder


from torchvision.datasets import OxfordIIITPet


from torchvision.datasets import StanfordCars


import torch.nn.functional as F


import numpy as np


from typing import List


from typing import Tuple


import pandas as pd


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim.lr_scheduler import LambdaLR


class SoftTargetCrossEntropy(torch.nn.Module):
    """Cross Entropy w/ smoothing or soft targets
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py
    """

    def __init__(self) ->None:
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SoftTargetCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_bwconrad_vit_finetune(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

