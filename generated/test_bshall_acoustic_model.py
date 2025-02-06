import sys
_module = sys.modules[__name__]
del sys
acoustic = _module
dataset = _module
model = _module
utils = _module
generate = _module
hubconf = _module
mels = _module
train = _module

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


import torch


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.nn.utils.rnn import pad_sequence


import torch.nn as nn


from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


import matplotlib


import torchaudio.transforms as transforms


import matplotlib.pylab as plt


import logging


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


import torch.distributed as dist


from torch.utils.data.distributed import DistributedSampler


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


class PreNet(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', output_size: 'int', dropout: 'float'=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, output_size), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.prenet = PreNet(128, 256, 256)
        self.lstm1 = nn.LSTM(512 + 256, 768, batch_first=True)
        self.lstm2 = nn.LSTM(768, 768, batch_first=True)
        self.lstm3 = nn.LSTM(768, 768, batch_first=True)
        self.proj = nn.Linear(768, 128, bias=False)

    def forward(self, x: 'torch.Tensor', mels: 'torch.Tensor') ->torch.Tensor:
        mels = self.prenet(mels)
        x, _ = self.lstm1(torch.cat((x, mels), dim=-1))
        res = x
        x, _ = self.lstm2(x)
        x = res + x
        res = x
        x, _ = self.lstm3(x)
        x = res + x
        return self.proj(x)

    @torch.inference_mode()
    def generate(self, xs: 'torch.Tensor') ->torch.Tensor:
        m = torch.zeros(xs.size(0), 128, device=xs.device)
        h1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        h2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        h3 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c3 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        mel = []
        for x in torch.unbind(xs, dim=1):
            m = self.prenet(m)
            x = torch.cat((x, m), dim=1).unsqueeze(1)
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1)


class Encoder(nn.Module):

    def __init__(self, discrete: 'bool'=False, upsample: 'bool'=True):
        super().__init__()
        self.embedding = nn.Embedding(100 + 1, 256) if discrete else None
        self.prenet = PreNet(256, 256, 256)
        self.convs = nn.Sequential(nn.Conv1d(256, 512, 5, 1, 2), nn.ReLU(), nn.InstanceNorm1d(512), nn.ConvTranspose1d(512, 512, 4, 2, 1) if upsample else nn.Identity(), nn.Conv1d(512, 512, 5, 1, 2), nn.ReLU(), nn.InstanceNorm1d(512), nn.Conv1d(512, 512, 5, 1, 2), nn.ReLU(), nn.InstanceNorm1d(512))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.prenet(x)
        x = self.convs(x.transpose(1, 2))
        return x.transpose(1, 2)


class AcousticModel(nn.Module):

    def __init__(self, discrete: 'bool'=False, upsample: 'bool'=True):
        super().__init__()
        self.encoder = Encoder(discrete, upsample)
        self.decoder = Decoder()

    def forward(self, x: 'torch.Tensor', mels: 'torch.Tensor') ->torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x, mels)

    @torch.inference_mode()
    def generate(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.encoder(x)
        return self.decoder.generate(x)


class LogMelSpectrogram(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160, center=False, power=1.0, norm='slaney', onesided=True, n_mels=128, mel_scale='slaney')

    def forward(self, wav):
        padding = (1024 - 160) // 2
        wav = F.pad(wav, (padding, padding), 'reflect')
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-05))
        return logmel


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PreNet,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_bshall_acoustic_model(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

