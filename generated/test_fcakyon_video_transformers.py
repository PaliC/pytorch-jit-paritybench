import sys
_module = sys.modules[__name__]
del sys
setup = _module
tests = _module
run_code_style = _module
test_auto_backbone = _module
test_auto_head = _module
test_auto_neck = _module
test_backbone = _module
test_onnx = _module
test_video_model = _module
utils = _module
video_transformers = _module
auto = _module
backbone = _module
head = _module
neck = _module
backbones = _module
base = _module
timm = _module
transformers = _module
data = _module
deployment = _module
gradio = _module
onnx = _module
heads = _module
hfhub_wrapper = _module
hub_mixin = _module
modeling = _module
necks = _module
predict = _module
pytorchvideo_wrapper = _module
labeled_video_dataset = _module
labeled_video_paths = _module
schedulers = _module
tasks = _module
single_label_classification = _module
templates = _module
tracking = _module
trainer = _module
extra = _module
file = _module
imports = _module
logger = _module

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


import inspect


from typing import Dict


from torch import nn


from typing import Tuple


import torch


import torch.utils.data


from torch.utils.data import DataLoader


from torchvision.transforms import CenterCrop


from torchvision.transforms import Compose


from torchvision.transforms import Lambda


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomHorizontalFlip


from typing import Optional


from typing import List


from typing import Union


import math


from collections import defaultdict


import logging


from typing import Any


from typing import Callable


from typing import Type


from typing import cast


from torchvision.datasets.folder import find_classes


from torchvision.datasets.folder import has_file_allowed_extension


from torchvision.datasets.folder import make_dataset


import numpy as np


from torch.optim.lr_scheduler import _LRScheduler


from typing import Generator


from typing import MutableMapping


from torch import Tensor


def class_to_config(class_, allowed_types: 'Tuple[Any]'=(int, float, str, dict, list, tuple, bool), ignored_attrs: 'Tuple[str]'=('config', 'dump_patches', 'training')):
    """
    Converts a class attributes into a config dict.

    Args:
        class_: The class to convert.
        allowed_types: The attribute value types that are allowed in the config.
        ignored_attrs: The attributes that are ignored.

    Returns:
        The config dict.
    """
    config = {'name': class_.__class__.__name__}
    for attribute in dir(class_):
        if attribute not in ignored_attrs:
            if not attribute.startswith('__') and not attribute.startswith('_'):
                value = getattr(class_, attribute)
                if type(value) in allowed_types:
                    if type(value) == tuple:
                        value = list(value)
                    config[attribute] = value
    return config


def get_num_total_params(model: 'nn.Module'):
    return sum(param.numel() for param in model.parameters())


def get_num_trainable_params(model: 'nn.Module'):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()

    def forward(self, x):
        raise NotImplementedError()

    def unfreeze_last_n_stages(self, n):
        raise NotImplementedError()

    @property
    def num_trainable_params(self):
        return get_num_trainable_params(self.model)

    @property
    def num_total_params(self):
        return get_num_total_params(self.model)

    @property
    def type(self) ->str:
        NotImplementedError()

    @property
    def framework(self) ->Dict:
        NotImplementedError()

    @property
    def model_name(self) ->str:
        NotImplementedError()

    @property
    def config(self) ->Dict:
        return class_to_config(self)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TimmBackbone(Backbone):

    def __init__(self, model_name: 'str', pretrained: 'bool'=False, num_unfrozen_stages=0, **backbone_kwargs):
        super(TimmBackbone, self).__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained, **backbone_kwargs)
        if backbone.pretrained_cfg['classifier'] == 'head.fc':
            backbone.head.fc = Identity()
        elif backbone.pretrained_cfg['classifier'] == 'fc':
            backbone.fc = Identity()
        elif backbone.pretrained_cfg['classifier'] == 'head':
            backbone.head = Identity()
        elif backbone.pretrained_cfg['classifier'] == 'classifier':
            backbone.classifier = Identity()
        elif backbone.pretrained_cfg['classifier'] == ('head.l', 'head_dist.l'):
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise NotImplementedError(f'Backbone not supported: {backbone.pretrained_cfg}')
        mean, std = backbone.pretrained_cfg['mean'], backbone.pretrained_cfg['std']
        num_features = backbone.num_features
        self.model = backbone
        self.num_features = num_features
        self.mean = mean
        self.std = std
        self._model_name = model_name
        self._type = '2d_backbone'
        self.unfreeze_last_n_stages(num_unfrozen_stages)

    @property
    def type(self):
        return self._type

    @property
    def framework(self):
        return {'name': 'timm', 'version': timm.__version__}

    @property
    def model_name(self):
        return self._model_name

    def forward(self, x):
        return self.model(x)

    def unfreeze_last_n_stages(self, n):
        stages = [stage for stage in self.model.stem.children()] + [stage for stage in self.model.stages.children()]
        unfreeze_last_n_stages_torch(stages, n)


models_2d = ['convnext', 'levit', 'cvt', 'clip', 'swin', 'vit', 'deit', 'beit', 'resnet']


models_3d = ['videomae', 'timesformer']


class TransformersBackbone(Backbone):

    def __init__(self, model_name: 'str', num_unfrozen_stages=0, **backbone_kwargs):
        super(TransformersBackbone, self).__init__()
        backbone = AutoModel.from_pretrained(model_name, **backbone_kwargs)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, **backbone_kwargs)
        if backbone.base_model_prefix == 'clip':
            backbone = CLIPVisionModel.from_pretrained(model_name)
        mean, std = feature_extractor.image_mean, feature_extractor.image_std
        if hasattr(backbone.config, 'hidden_size'):
            num_features = backbone.config.hidden_size
        elif hasattr(backbone.config, 'hidden_sizes'):
            num_features = backbone.config.hidden_sizes[-1]
        elif hasattr(backbone.config, 'embed_dim'):
            num_features = backbone.config.embed_dim[-1]
        elif hasattr(backbone.config, 'projection_dim'):
            num_features = backbone.config.projection_dim
        else:
            raise NotImplementedError(f'Huggingface model not supported: {backbone.base_model_prefix}')
        if hasattr(backbone.config, 'num_frames'):
            self._num_frames = backbone.config.num_frames
        else:
            self._num_frames = 1
        self.model = backbone
        self.num_features = num_features
        self.mean = mean
        self.std = std
        self._model_name = model_name
        if self.model.base_model_prefix in models_2d:
            self._type = '2d_backbone'
        elif self.model.base_model_prefix in models_3d:
            self._type = '3d_backbone'
        else:
            raise NotImplementedError(f'Huggingface model not supported: {self.model.base_model_prefix}')
        self.unfreeze_last_n_stages(num_unfrozen_stages)

    @property
    def type(self) ->str:
        return self._type

    @property
    def framework(self) ->Dict:
        return {'name': 'transformers', 'version': transformers.__version__}

    @property
    def model_name(self) ->str:
        return self._model_name

    @property
    def num_frames(self) ->int:
        return self._num_frames

    def forward(self, x):
        if self.model.base_model_prefix in models_3d:
            if x.shape[2] != self.num_frames:
                raise ValueError(f'Input has {x.shape[2]} frames, but {self.model_name} accepts {self.num_frames} frames. Set num_timesteps to {self.num_frames}.')
            x = x.permute(0, 2, 1, 3, 4)
            output = self.model(pixel_values=x, return_dict=True)[0]
        else:
            output = self.model(pixel_values=x, return_dict=True)[1]
        if output.dim() == 3:
            output = output.mean(1)
        return output

    def unfreeze_last_n_stages(self, n):
        if self.model.base_model_prefix == 'convnext':
            stages = []
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            stages.extend(self.model.base_model.encoder.stages)
            stages.append(self.model.base_model.layernorm)
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == 'levit':
            stages = []
            for param in self.model.base_model.patch_embeddings.parameters():
                param.requires_grad = False
            stages.extend(self.model.base_model.encoder.stages)
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == 'cvt':
            stages = []
            stages.extend(list(self.model.base_model.encoder.stages.children()))
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == 'clip':
            stages = []
            for param in self.model.base_model.vision_model.embeddings.parameters():
                param.requires_grad = False
            stages.extend(list(self.model.base_model.vision_model.encoder.layers.children()))
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix in ['swin', 'vit', 'deit', 'beit']:
            stages = []
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            stages.extend(list(self.model.base_model.encoder.layers.children()))
            stages.append(self.model.base_model.layernorm)
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == 'videomae':
            stages = []
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            stages.extend(list(self.model.base_model.encoder.layer.children()))
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == 'timesformer':
            stages = []
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            stages.extend(list(self.model.base_model.encoder.layer.children()))
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == 'resnet':
            stages = []
            for param in self.model.base_model.embedder.parameters():
                param.requires_grad = False
            stages.extend(self.model.base_model.encoder.stages)
            unfreeze_last_n_stages_torch(stages, n)
        else:
            raise NotImplementedError(f'Freezing not supported for Huggingface model: {self.model.base_model_prefix}')


class LinearHead(nn.Module):
    """
     (BxF)
       ↓
    Dropout
       ↓
    Linear
    """

    def __init__(self, hidden_size: 'int', num_classes: 'int', dropout_p: 'float'=0.0):
        super(LinearHead, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p) if dropout_p != 0 else None
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        return x

    @property
    def config(self) ->Dict:
        return class_to_config(self)


class TimeDistributed(nn.Module):
    """
    Time x Backbone2D (BxCxTxHxW)
           ↓
         Pool2D
           ↓
        (BxTxF)
    """

    def __init__(self, backbone: 'video_transformers.backbones.base.Backbone', low_memory=False):
        super(TimeDistributed, self).__init__()
        self.backbone = backbone
        self.low_memory = low_memory

    @property
    def num_features(self):
        return self.backbone.num_features

    @property
    def mean(self):
        return self.backbone.mean

    @property
    def std(self):
        return self.backbone.std

    @property
    def type(self):
        return self.backbone.type

    @property
    def model_name(self):
        return self.backbone.model_name

    @property
    def framework(self):
        return self.backbone.framework

    @property
    def num_trainable_params(self):
        return self.backbone.num_trainable_params

    @property
    def num_total_params(self):
        return self.backbone.num_total_params

    @property
    def config(self):
        return self.backbone.config

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        batch_size, num_channels, num_timesteps, height, width = x.size()
        if self.low_memory:
            output_matrix = []
            for timestep in range(num_timesteps):
                x_t = self.backbone(x[:, :, timestep, :, :])
                output_matrix.append(x_t)
            x = torch.stack(output_matrix, dim=1)
            x_t = None
            output_matrix = None
        else:
            x = x.permute((0, 2, 1, 3, 4))
            x = x.contiguous().view(batch_size * num_timesteps, num_channels, height, width)
            x = self.backbone(x)
            x = x.contiguous().view(batch_size, num_timesteps, x.size(1))
        return x


class BaseNeck(nn.Module):

    @property
    def config(self) ->Dict:
        return class_to_config(self)


class LSTMNeck(BaseNeck):
    """
        (BxTxF)
           ↓
         LSTM
           ↓
    (BxF) or (BxTxF)
    """

    def __init__(self, num_features, hidden_size, num_layers, return_last=True):
        """
        Create a LSTMNeck.

        Args:
            num_features: Number of input features.
            hidden_size: Number of hidden units.
            num_layers: Number of layers.
            return_last: If True, return the last hidden state of the LSTM.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_last = return_last
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x, _ = self.lstm(x)
        if self.return_last:
            x = x[:, -1, :]
        return x


class GRUNeck(BaseNeck):
    """
        (BxTxF)
           ↓
          GRU
           ↓
    (BxF) or (BxTxF)
    """

    def __init__(self, num_features: 'int', hidden_size: 'int', num_layers: 'int', return_last: 'bool'=True):
        """
        Create a GRUNeck.

        Args:
            num_features: Number of input features.
            hidden_size: Number of hidden units.
            num_layers: Number of layers.
            return_last: If True, return the last hidden state of the GRU.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_last = return_last
        self.gru = nn.GRU(num_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x, _ = self.gru(x)
        if self.return_last:
            x = x[:, -1, :]
        return x


class PostionalEncoder(nn.Module):
    """
         (BxTxF)
            ↓
    PostionalEncoder
            ↓
         (BxTxF)
    """

    def __init__(self, num_features: 'int', dropout_p: 'float'=0.0, num_timesteps: 'int'=30):
        super(PostionalEncoder, self).__init__()
        self.num_features = num_features
        self.dropout_p = dropout_p
        self.num_timesteps = num_timesteps
        self.dropout = nn.Dropout(dropout_p) if dropout_p != 0 else None
        self.scale_constat = torch.sqrt(torch.tensor(self.num_features))
        position_encodings = torch.zeros(self.num_timesteps, self.num_features)
        for time_ind in range(self.num_timesteps):
            for feat_ind in range(0, self.num_features, 2):
                sin_input = time_ind / 10000 ** (2 * feat_ind / self.num_features)
                cos_input = time_ind / 10000 ** (2 * (feat_ind + 1) / self.num_features)
                position_encodings[time_ind, feat_ind] = torch.sin(torch.tensor(sin_input))
                position_encodings[time_ind, feat_ind + 1] = torch.cos(torch.tensor(cos_input))
        self.position_encodings = position_encodings

    def add_positional_encoding(self, x: 'torch.Tensor') ->torch.Tensor:
        self.position_encodings = self.position_encodings
        batch_size = x.size(0)
        x = x + self.position_encodings.repeat(batch_size, 1, 1)
        return x

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x * self.scale_constat
        x = self.add_positional_encoding(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class TransformerNeck(BaseNeck):
    """
        (BxTxF)
           ↓
      Transformer
           ↓
    (BxF) or (BxTxF)
    """

    def __init__(self, num_features: 'int', num_timesteps: 'int', transformer_enc_num_heads: 'int'=4, transformer_enc_num_layers: 'int'=2, transformer_enc_act: 'int'='gelu', dropout_p: 'int'=0.0, return_mean: 'bool'=True):
        """
        Create a TransformerNeck.

        Args:
            num_features: number of input features
            num_timesteps: number of timesteps
            transformer_enc_num_heads: number of heads in the transformer encoder
            transformer_enc_num_layers: number of layers in the transformer encoder
            transformer_enc_act: activation function for the transformer encoder
            dropout_p: dropout probability
            return_mean: return the mean of the transformed features
        """
        super(TransformerNeck, self).__init__()
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.transformer_enc_num_heads = transformer_enc_num_heads
        self.transformer_enc_num_layers = transformer_enc_num_layers
        self.transformer_enc_act = transformer_enc_act
        self.dropout_p = dropout_p
        self.return_mean = return_mean
        self.positional_encoder = PostionalEncoder(num_features, dropout_p, num_timesteps)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=transformer_enc_num_heads, activation=transformer_enc_act)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_enc_num_layers)

    def forward(self, x):
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        if self.return_mean:
            x = x.mean(dim=1)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GRUNeck,
     lambda: ([], {'num_features': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSTMNeck,
     lambda: ([], {'num_features': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LinearHead,
     lambda: ([], {'hidden_size': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PostionalEncoder,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 30, 4])], {}),
     True),
    (TransformerNeck,
     lambda: ([], {'num_features': 4, 'num_timesteps': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_fcakyon_video_transformers(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

