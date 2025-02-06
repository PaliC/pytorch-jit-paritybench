import sys
_module = sys.modules[__name__]
del sys
main = _module

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


import random


import re


import torch


import logging


import numpy as np


import time


from typing import Optional


from typing import Dict


from typing import Any


from typing import Tuple


from typing import List


from typing import Union


from typing import Set


from collections import defaultdict


from collections import deque


from logging.handlers import RotatingFileHandler


import torch.nn as nn


import torch.nn.functional as F


class ModalityFusionWithAttention(nn.Module):

    def __init__(self, modality_dims, fused_dim, hidden_dim=64):
        super().__init__()
        self.transformers = nn.ModuleList([nn.Linear(dim, fused_dim) for dim in modality_dims])
        self.attention_fc1 = nn.Linear(fused_dim, hidden_dim)
        self.attention_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, modality_features):
        transformed = []
        for i, feature in enumerate(modality_features):
            transformed.append(self.transformers[i](feature))
        stack = torch.stack(transformed, dim=1)
        attn_scores = self.attention_fc2(F.relu(self.attention_fc1(stack)))
        attn_scores = attn_scores.squeeze(-1)
        weights = torch.softmax(attn_scores, dim=1)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


class ContentEncoder(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=384):
        super(ContentEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.norm3 = nn.LayerNorm(self.output_dim)
        self.attention = nn.Linear(self.output_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.attention.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.norm3(x)
        attn = torch.sigmoid(self.attention(x))
        x = x * attn
        return x


class GenreEncoder(nn.Module):

    def __init__(self, input_dim=35, hidden_dim=128, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


class TemporalEncoder(nn.Module):

    def __init__(self, input_dim=32, hidden_dim=128, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


class MetadataEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_dim = 768
        self.hidden_dim = 512
        self.output_dim = 256
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.norm2 = nn.LayerNorm(self.output_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


class GraphConvNetwork(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=341):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x


class GraphAttentionNetwork(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=341, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = GATConv(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x


class GraphSageNetwork(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=342):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x

