import sys
_module = sys.modules[__name__]
del sys
in2n = _module
in2n = _module
in2n_config = _module
in2n_datamanager = _module
in2n_pipeline = _module
in2n_trainer = _module
ip2p = _module
clip_metrics = _module

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


from typing import Type


import torch


from itertools import cycle


from typing import Optional


from torch.cuda.amp.grad_scaler import GradScaler


from typing import Union


from torch import Tensor


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


CONST_SCALE = 0.18215


DDIM_SOURCE = 'CompVis/stable-diffusion-v1-4'


IP2P_SOURCE = 'timbrooks/instruct-pix2pix'


class ClipSimilarity(nn.Module):

    def __init__(self, name: 'str'='ViT-L/14'):
        super().__init__()
        assert name in ('RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px')
        self.size = {'RN50x4': 288, 'RN50x16': 384, 'RN50x64': 448, 'ViT-L/14@336px': 336}.get(name, 224)
        self.model, _ = clip.load(name, device='cpu', download_root='./')
        self.model.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer('std', torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text):
        text = clip.tokenize(text, truncate=True)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):
        image = F.interpolate(image.float(), size=self.size, mode='bicubic', align_corners=False)
        image = image - rearrange(self.mean, 'c -> 1 c 1 1')
        image = image / rearrange(self.std, 'c -> 1 c 1 1')
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(self, image_0, image_1, text_0, text_1):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image

