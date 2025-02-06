import sys
_module = sys.modules[__name__]
del sys
conftest = _module
stateful_example = _module
async_checkpoint_writing = _module
checkpoint_manual_save = _module
checkpoint_reading = _module
checkpoint_writing = _module
s3torchbenchmarking = _module
benchmark_utils = _module
constants = _module
datagen = _module
dataset = _module
benchmark = _module
dcp = _module
benchmark = _module
hydra_callback = _module
lightning_checkpointing = _module
benchmark = _module
checkpoint_profiler = _module
sample_counter = _module
models = _module
pytorch_checkpointing = _module
benchmark = _module
test_compatibility = _module
collect_and_write_to_dynamodb = _module
download_and_transform_results = _module
html_result_generator = _module
conf = _module
s3torchconnector = _module
_s3_bucket_iterable = _module
_s3bucket_key_data = _module
_s3client = _module
_mock_s3client = _module
s3client_config = _module
_s3dataset_common = _module
_user_agent = _module
_version = _module
s3_file_system = _module
lightning = _module
s3_lightning_checkpoint = _module
s3checkpoint = _module
s3iterable_dataset = _module
s3map_dataset = _module
s3reader = _module
s3writer = _module
test_e2e_s3_file_system = _module
lightning_transformer = _module
net = _module
test_common = _module
test_distributed_training = _module
test_e2e_s3_lightning_checkpoint = _module
test_e2e_s3checkpoint = _module
test_e2e_s3datasets = _module
test_mountpoint_client_parallel_access = _module
test_multiprocess_dataloading = _module
unit = _module
_checkpoint_byteorder_patch = _module
_hypothesis_python_primitives = _module
test_s3_file_system = _module
test_s3_lightning_checkpoint = _module
test_checkpointing = _module
test_lightning_missing = _module
test_s3_client = _module
test_s3_client_config = _module
test_s3dataset_common = _module
test_s3iterable_dataset = _module
test_s3mapdataset = _module
test_s3reader = _module
test_s3writer = _module
test_user_agent = _module
test_version = _module
s3torchconnectorclient = _module
_logger_patch = _module
test_logging = _module
test_mountpoint_s3_integration = _module
test_mountpoint_s3_client = _module
test_s3exception = _module
test_structs = _module

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


import torch.distributed as dist


import torch.distributed.checkpoint as dcp


import torch.multiprocessing as mp


import torch.nn as nn


from torch.distributed.checkpoint.state_dict import _patch_model_state_dict


from torch.distributed.checkpoint.state_dict import _patch_optimizer_state_dict


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.utils.data import DataLoader


import random


import string


import time


from collections import defaultdict


from collections import deque


from typing import Dict


from typing import Optional


from typing import List


from typing import TypedDict


import numpy as np


import torch.cuda


from torchvision.transforms import v2


from torch.utils.data import Dataset


from torch.utils.data import default_collate


import logging


from time import perf_counter


from typing import Tuple


import pandas as pd


from torch import multiprocessing as mp


from torch.distributed.checkpoint import FileSystemWriter


from torch.nn import Module


from torch.nn.parallel import DistributedDataParallel


from typing import Any


from typing import Union


from abc import ABC


from abc import abstractmethod


from functools import cached_property


from typing import Callable


from torch.utils.data.dataloader import DataLoader


from typing import Generator


from torch.distributed.checkpoint.filesystem import FileSystemReader


from torch.distributed.checkpoint.filesystem import FileSystemWriter


from torch.distributed.checkpoint.filesystem import FileSystemBase


from functools import partial


from typing import Iterator


from typing import Iterable


import torch.utils.data


from torch.distributed.checkpoint import CheckpointException


import torch.nn.functional as F


from torch import nn


from collections import Counter


from itertools import product


from typing import TYPE_CHECKING


from torch.utils.data import DistributedSampler


from torch.utils.data.datapipes.datapipe import MapDataPipe


from torch.utils.data import get_worker_info


from typing import Sequence


class Model(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device='cuda')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def equals(self, other_model: 'nn.Module') ->bool:
        for key_item_1, key_item_2 in zip(self.state_dict().items(), other_model.state_dict().items()):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                return False
        return True


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
]

class Test_awslabs_s3_connector_for_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

