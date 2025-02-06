import sys
_module = sys.modules[__name__]
del sys
__about__ = _module
aim = _module
__version__ = _module
acme = _module
catboost = _module
cli = _module
configs = _module
convert = _module
commands = _module
processors = _module
mlflow = _module
tensorboard = _module
wandb = _module
init = _module
manager = _module
runs = _module
utils = _module
server = _module
storage = _module
up = _module
version = _module
watcher_cli = _module
ext = _module
cleanup = _module
exception_resistant = _module
notebook = _module
notifier = _module
base_notifier = _module
config = _module
logging_notifier = _module
notifier_builder = _module
slack_notifier = _module
workplace_notifier = _module
pynvml = _module
resource = _module
log = _module
stat = _module
tracker = _module
sshfs = _module
task_queue = _module
queue = _module
tensorboard_tracker = _module
run = _module
tracker = _module
transport = _module
client = _module
handlers = _module
heartbeat = _module
message_utils = _module
remote_resource = _module
request_queue = _module
router = _module
tracking = _module
fastai = _module
hf_dataset = _module
hugging_face = _module
keras = _module
keras_tuner = _module
lightgbm = _module
mxnet = _module
optuna = _module
paddle = _module
prophet = _module
pytorch = _module
pytorch_ignite = _module
pytorch_lightning = _module
sb3 = _module
sdk = _module
adapters = _module
keras_mixins = _module
pytorch_ignite = _module
tensorflow = _module
xgboost = _module
base_run = _module
callbacks = _module
caller = _module
events = _module
helpers = _module
data_version = _module
errors = _module
index_manager = _module
legacy = _module
deprecation_warning = _module
flush = _module
select = _module
session = _module
track = _module
lock_manager = _module
logging = _module
log_record = _module
maintenance_run = _module
num_utils = _module
objects = _module
artifact = _module
audio = _module
distribution = _module
figure = _module
image = _module
io = _module
wavfile = _module
plugins = _module
deeplake_dataset = _module
dvc_metadata = _module
hf_datasets_metadata = _module
hub_dataset = _module
text = _module
query_utils = _module
remote_repo_proxy = _module
remote_run_reporter = _module
repo = _module
repo_utils = _module
reporter = _module
file_manager = _module
run_status_watcher = _module
sequence = _module
sequence_collection = _module
sequences = _module
audio_sequence = _module
distribution_sequence = _module
figure_sequence = _module
image_sequence = _module
metric = _module
sequence_type_map = _module
text_sequence = _module
training_flow = _module
types = _module
uri_service = _module
arrayview = _module
artifacts = _module
artifact_registry = _module
artifact_storage = _module
filesystem_storage = _module
s3_storage = _module
container = _module
containertreeview = _module
context = _module
encoding = _module
env = _module
hashing = _module
inmemorytreeview = _module
lock_proxy = _module
locking = _module
migrations = _module
b07e7b07c8ce_ = _module
fbfe5c4702fb_soft_delete = _module
object = _module
prefixview = _module
proxy = _module
query = _module
structured = _module
db = _module
entities = _module
sql_engine = _module
factory = _module
models = _module
treearrayview = _module
treeutils_non_native = _module
treeview = _module
treeviewproxy = _module
deprecation = _module
web = _module
api = _module
dashboard_apps = _module
pydantic_models = _module
serializers = _module
views = _module
dashboards = _module
experiments = _module
projects = _module
project = _module
reports = _module
object_api_utils = _module
object_views = _module
tags = _module
middlewares = _module
profiler = _module
aim_ui = _module
aim_ui_core = _module
setup = _module
conf = _module
acme_track = _module
catboost_track = _module
fastai_track = _module
hugging_face_track = _module
keras_track = _module
keras_tuner_track = _module
lightgbm_track = _module
mxnet_track = _module
optuna_track = _module
paddle_track = _module
prophet_track = _module
pytorch_ignite_track = _module
pytorch_lightning_track = _module
pytorch_track = _module
pytorch_track_images = _module
sb3_track = _module
tensorboard_aim_sync = _module
tensorflow_keras_track = _module
xgboost_track = _module
main = _module
performance_tests = _module
base = _module
conftest = _module
queries = _module
test_data_collection = _module
test_query = _module
test_container_open = _module
test_iterative_access = _module
test_random_access = _module
tests = _module
test_dashboards_api = _module
test_project_api = _module
test_run_api = _module
test_run_images_api = _module
test_structured_data_api = _module
test_tensorboard_run = _module
test_tensorboard_tracker = _module
integrations = _module
test_deeplake_dataset = _module
test_dvc_metadata = _module
test_hf_datasets = _module
test_hub_dataset = _module
test_image_construction = _module
test_resource_tracker = _module
test_run_apis = _module
test_run_creation_checks = _module
test_run_finalization_time = _module
test_run_metric_types = _module
test_run_track_type_checking = _module
test_run_write_container_data = _module
test_utils = _module
test_query_with_epoch_none = _module
test_structured_db = _module
troubleshooting = _module
base_project_statistics = _module

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


import queue


import time


from typing import Any


import tensorflow as tf


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


from itertools import chain


from itertools import repeat


import numpy as np


import torch


import torch.nn.functional as F


from torch import nn


from torch import optim


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


from torch import utils


from torchvision.datasets import MNIST


from torchvision.transforms import ToTensor


import torch.nn as nn


import torchvision


import torchvision.transforms as transforms


from queue import Queue


from torch.utils.tensorboard.summary import histogram


from torch.utils.tensorboard.summary import histogram_raw


from torch.utils.tensorboard.summary import image


from torch.utils.tensorboard.summary import scalar


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.convlayer1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.convlayer2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_aimhubio_aim(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

