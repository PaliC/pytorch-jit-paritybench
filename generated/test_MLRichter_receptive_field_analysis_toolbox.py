import sys
_module = sys.modules[__name__]
del sys
conf = _module
rfa_toolbox = _module
architectures = _module
resnet = _module
vgg = _module
domain = _module
encodings = _module
pytorch = _module
domain = _module
ingest_architecture = _module
intermediate_graph = _module
layer_handlers = _module
substitutors = _module
utils = _module
tensorflow_keras = _module
graphs = _module
test_graphviz = _module
graph_utils = _module
vizualize = _module
tests = _module
test_encodings = _module
test_pytorch = _module
test_tensorflow = _module
test_graph = _module
test_dynamic_graph = _module
test_utils = _module
test_viz = _module

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


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import warnings


import numpy as np


from typing import Sequence


from torchvision.models import efficientnet_b0


from torchvision.models.alexnet import alexnet


from torchvision.models.inception import inception_v3


from torchvision.models.mnasnet import mnasnet1_3


from torchvision.models.resnet import resnet18


from torchvision.models.resnet import resnet152


from torchvision.models.vgg import vgg19


class SomeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.k_size = 3
        self.s_size = 1
        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=self.k_size, stride=self.s_size, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=self.k_size * 2, stride=self.s_size * 2, padding=2)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SomeModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
]

class Test_MLRichter_receptive_field_analysis_toolbox(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

