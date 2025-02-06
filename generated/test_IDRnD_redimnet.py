import sys
_module = sys.modules[__name__]
del sys
hubconf = _module
redimnet = _module
attention = _module
convnext = _module
features = _module
features_tf = _module
layernorm = _module
poolings = _module
redim_structural = _module
resblocks = _module
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


import math


import functools


import numpy as np


import torch.nn as nn


from typing import List


from torch import Tensor


import torch.nn.functional as F


from collections import OrderedDict


from typing import Iterable


from typing import Optional


import torchaudio


from scipy.signal import windows


class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.0, bias: 'bool'=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class NewGELUActivation(nn.Module):

    def forward(self, input: 'Tensor') ->Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):

    def __init__(self, use_gelu_python: 'bool'=False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = F.gelu

    def _gelu_python(self, input: 'Tensor') ->Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: 'Tensor') ->Tensor:
        return self.act(input)


def gelu(x):
    """This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ClippedGELUActivation(nn.Module):

    def __init__(self, min: 'float', max: 'float'):
        if min > max:
            raise ValueError(f'min should be < max (got min: {min}, max: {max})')
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: 'Tensor') ->Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class FastGELUActivation(nn.Module):

    def forward(self, input: 'Tensor') ->Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):

    def forward(self, input: 'Tensor') ->Tensor:
        return input * torch.sigmoid(1.702 * input)


class SiLUActivation(nn.Module):

    def forward(self, input: 'Tensor') ->Tensor:
        return F.silu(input)


class MishActivation(nn.Module):

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse('1.9.0'):
            self.act = self._mish_python
        else:
            self.act = F.mish

    def _mish_python(self, input: 'Tensor') ->Tensor:
        return input * torch.tanh(F.softplus(input))

    def forward(self, input: 'Tensor') ->Tensor:
        return self.act(input)


class LinearActivation(nn.Module):

    def forward(self, input: 'Tensor') ->Tensor:
        return input


ACT2CLS = {'gelu': GELUActivation, 'gelu_10': (ClippedGELUActivation, {'min': -10, 'max': 10}), 'gelu_fast': FastGELUActivation, 'gelu_new': NewGELUActivation, 'gelu_python': (GELUActivation, {'use_gelu_python': True}), 'linear': LinearActivation, 'mish': MishActivation, 'quick_gelu': QuickGELUActivation, 'relu': nn.ReLU, 'relu6': nn.ReLU6, 'sigmoid': nn.Sigmoid, 'silu': SiLUActivation, 'swish': SiLUActivation, 'tanh': nn.Tanh}


class ClassInstantier(OrderedDict):

    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2FN = ClassInstantier(ACT2CLS)


class FeedForward(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', hidden_act: 'str'='gelu_new', activation_dropout: 'float'=0.0, hidden_dropout: 'float'=0.0):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act]
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class TransformerEncoderLayer(nn.Module):

    def __init__(self, n_state: 'int', n_mlp: 'int', n_head: 'int', channel_last: 'bool'=False, act: 'str'='gelu_new', act_do: 'float'=0.0, att_do: 'float'=0.0, hid_do: 'float'=0.0, ln_eps: 'float'=1e-06):
        hidden_size = n_state
        num_attention_heads = n_head
        intermediate_size = n_mlp
        hidden_act = act
        activation_dropout = act_do
        attention_dropout = att_do
        hidden_dropout = hid_do
        layer_norm_eps = ln_eps
        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(embed_dim=hidden_size, num_heads=num_attention_heads, dropout=attention_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size=hidden_size, hidden_act=hidden_act, intermediate_size=intermediate_size, activation_dropout=activation_dropout, hidden_dropout=hidden_dropout)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs


BatchNormNd = {(1): nn.BatchNorm1d, (2): nn.BatchNorm2d}


ConvNd = {(1): nn.Conv1d, (2): nn.Conv2d}


class ConvNeXtLikeBlock(nn.Module):

    def __init__(self, C, dim=2, kernel_sizes=[(3, 3)], Gdiv=1, padding='same', activation='gelu'):
        super().__init__()
        self.dwconvs = nn.ModuleList(modules=[ConvNd[dim](C, C, kernel_size=ks, padding=padding, groups=C // Gdiv if Gdiv is not None else 1) for ks in kernel_sizes])
        self.norm = BatchNormNd[dim](C * len(kernel_sizes))
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        self.pwconv1 = ConvNd[dim](C * len(kernel_sizes), C, 1)

    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs], dim=1)
        x = self.act(self.norm(x))
        x = self.pwconv1(x)
        x = skip + x
        return x


class NormalizeAudio(nn.Module):

    def __init__(self, eps: 'float'=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return ((x - x.mean(dim=2, keepdims=True)) / (x.std(dim=2, keepdims=True, unbiased=False) + self.eps)).squeeze(1)


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: 'float'=0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10), freq_start_bin=0):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.freq_start_bin = freq_start_bin
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width
        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(self.freq_start_bin, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < mask_pos + mask_len)
        mask = mask.any(dim=1)
        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class MelBanks(nn.Module):

    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600, n_mels=80, do_spec_aug=False, norm_signal=False, do_preemph=True, spec_norm='mn', freq_start_bin=0, num_apply_spec_aug=1, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        super(MelBanks, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.torchfbank = torch.nn.Sequential(NormalizeAudio() if norm_signal else nn.Identity(), PreEmphasis() if do_preemph else nn.Identity(), torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, f_min=f_min, f_max=f_max, n_mels=n_mels, window_fn=torch.hamming_window))
        self.spec_norm = spec_norm
        if spec_norm == 'mn':
            self.spec_norm = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
        elif spec_norm == 'mvn':
            self.spec_norm = lambda x: (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-08)
        elif spec_norm == 'bn':
            self.spec_norm = nn.BatchNorm1d(n_mels)
        else:
            pass
        if do_spec_aug:
            self.specaug = FbankAug(freq_start_bin=freq_start_bin, freq_mask_width=freq_mask_width, time_mask_width=time_mask_width)
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.amp.autocast(enabled=False):
                x = self.torchfbank(x) + 1e-06
                x = x.log()
                x = self.spec_norm(x)
                if self.training:
                    for _ in range(self.num_apply_spec_aug):
                        x = self.specaug(x)
        return x


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.0)


def get_filterbanks(low_freq: 'int'=20, high_freq: 'int'=7600, nfilt: 'int'=80, nfft: 'int'=512, samplerate: 'int'=16000):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param low_freq: lowest band edge of mel filters, default 0 Hz
    :param high_freq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    lowmel = hz2mel(low_freq)
    highmel = hz2mel(high_freq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    lower_edge_mel = melpoints[:-2].reshape(1, -1)
    center_mel = melpoints[1:-1].reshape(1, -1)
    upper_edge_mel = melpoints[2:].reshape(1, -1)
    spectrogram_bins_mel = hz2mel(np.linspace(0, samplerate // 2, nfft))[1:].reshape(-1, 1)
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.vstack([np.zeros((1, nfilt)), mel_weights_matrix])[:, :].astype('float32')


class SpectralFeaturesTF(nn.Module):

    def __init__(self, frame_length: 'int'=400, frame_step: 'int'=160, fft_length: 'int'=512, sample_rate: 'int'=16000, window: 'str'='hann', normalize_spectrogram: 'bool'=False, normalize_signal: 'bool'=False, eps: 'float'=1e-08, mode: 'str'='melbanks', low_freq: 'int'=20, high_freq: 'int'=7600, num_bins: 'int'=80, log_mels: 'bool'=True, fft_mode: 'str'='abs', sqrt_real_imag: 'bool'=False, return_img: 'bool'=False, **kwargs):
        """
        Requirements
        ------------
        input shape must meet the conditions: mod((input.shape[0] - length), shift) == 0
        fft_length >= frame_length

        Parameters
        ------------
        :param frame_length: Length of each segment in # of samples
        :param frame_step: Shift between segments in # of samples
        :param fft_length: number of dft points, if None => fft_length == frame_length
        :param fft_mode: "abs" - amplitude spectrum; "real" - only real part, "imag" - only imag part,
        "complex" - concatenate real and imag part.
        :param kwargs: unuse

        Input
        -----
        input mut have shape: [n_batch, signal_length, 1]

        Returns
        -------
        A keras model that has output shape of
        (None, nfft / 2, n_time) (if type == "abs" || "real" || "imag") or
        (None, nfft / 2, n_frame, 2) (if type = "abs" & `img_dim_ordering() == 'tf').
        (None, nfft / 2, n_frame, 2) (if type = "complex" & `img_dim_ordering() == 'tf').

        number of time point of output spectrogram: n_time = (input.shape[0] - length) / shift + 1
        """
        super().__init__()
        assert mode in ['fft', 'melbanks', 'mfcc', 'complex']
        assert isinstance(frame_length, int) and isinstance(frame_step, int) and isinstance(fft_length, int)
        self.length = frame_length
        self.shift = frame_step
        self.sqrt_real_imag = sqrt_real_imag
        self.normalize_spectrogram = normalize_spectrogram
        self.normalize_signal = normalize_signal
        self.window = window
        self.eps = eps
        if fft_length is None:
            self.nfft = frame_length
        else:
            self.nfft = fft_length
        self.samplerate = sample_rate
        self.features = mode
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_bins = num_bins
        self.return_img = return_img
        if mode in ['melbanks', 'mfcc']:
            fft_mode = 'abs'
        self.fft_mode = fft_mode
        self.log_mels = log_mels
        self.build()

    def build(self):
        assert self.nfft >= self.length
        if self.window:
            if self.window == 'hamming':
                self.window = windows.hamming(self.length)
            elif self.window in ['hann', 'hanning']:
                self.window = np.array([(0.5 - 0.5 * np.cos(2 * np.pi * l / (self.length - 1))) for l in range(self.length)])
            elif self.window == 'sqrt_hann':
                self.window = np.array([(0.5 - 0.5 * np.cos(2 * np.pi * l / (self.length - 1))) for l in range(self.length)]) ** 0.5
            elif self.window == 'kaiser':
                self.window = windows.kaiser(self.length)
            else:
                self.window = np.ones(self.length)
        self.window = self.window.astype('float32')
        real_kernel = np.asarray([np.cos(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft) for n in range(self.nfft)]).astype('float32').T
        self.real_kernel = real_kernel[:self.length, :self.nfft // 2]
        if self.window is not None:
            self.real_kernel *= self.window[:, None]
        self.real_kernel = self.real_kernel[:, None, :]
        image_kernel = np.asarray([np.sin(2 * np.pi * np.arange(0, self.nfft) * n / self.nfft) for n in range(self.nfft)]).astype('float32').T
        self.image_kernel = image_kernel[:self.length, :self.nfft // 2]
        if self.window is not None:
            self.image_kernel *= self.window[:, None]
        self.image_kernel = self.image_kernel[:, None, :]
        self.register_buffer('real_kernel_pt', torch.from_numpy(self.real_kernel).permute(2, 1, 0).float())
        self.register_buffer('image_kernel_pt', torch.from_numpy(self.image_kernel).permute(2, 1, 0).float())
        if self.features in ['melbanks']:
            linear_to_mel_weight_matrix = get_filterbanks(nfilt=self.num_bins, nfft=self.nfft // 2, samplerate=self.samplerate, low_freq=self.low_freq, high_freq=self.high_freq)
            linear_to_mel_weight_matrix = linear_to_mel_weight_matrix[:, :, None]
            self.register_buffer('melbanks_pt', torch.from_numpy(linear_to_mel_weight_matrix).permute(1, 0, 2).float())

    def forward(self, inputs):
        dtype = inputs.dtype
        inputs = inputs.float()
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if self.normalize_signal:
            inputs = (inputs - inputs.mean(dim=2, keepdims=True)) / (inputs.std(dim=2, keepdims=True, unbiased=False) + self.eps)
        real_part = F.conv1d(inputs, self.real_kernel_pt, stride=self.shift, padding=self.shift // 2)
        imag_part = F.conv1d(inputs, self.image_kernel_pt, stride=self.shift, padding=self.shift // 2)
        if self.features == 'complex':
            return [real_part, imag_part]
        fft = torch.square(real_part) + torch.square(imag_part)
        if self.sqrt_real_imag:
            fft = torch.sqrt(fft)
        feat = fft.clip(self.eps, 1 / self.eps)
        if self.fft_mode == 'log':
            feat = torch.log(feat)
        if self.features in ['melbanks']:
            mel_spectrograms = F.conv1d(feat, self.melbanks_pt, stride=1, padding=0)
            mel_spectrograms = mel_spectrograms.clip(self.eps, 1 / self.eps)
            if self.log_mels:
                feat = torch.log(mel_spectrograms)
            else:
                feat = mel_spectrograms
        if self.normalize_spectrogram:
            feat = (feat - feat.mean(dim=(1, 2), keepdims=True)) / (feat.std(dim=(1, 2), keepdims=True, unbiased=False) + self.eps)
        if self.return_img:
            feat = feat[:, None, :, :]
        return feat


class LogSpec(nn.Module):

    def __init__(self, eps: 'float'=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x.clip(self.eps, 100000000.0).log()


class TFMelBanks(nn.Module):

    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600, n_mels=80, do_spec_aug=False, norm_signal=False, do_preemph=True, freq_start_bin=0, freq_mask_width=(0, 8), time_mask_width=(0, 10), eps=1e-08):
        super(TFMelBanks, self).__init__()
        self.torchfbank = torch.nn.Sequential(NormalizeAudio(eps) if norm_signal else nn.Identity(), PreEmphasis() if do_preemph else nn.Identity(), SpectralFeaturesTF(frame_length=win_length, frame_step=hop_length, fft_length=n_fft, sample_rate=sample_rate, window='hamming', normalize_spectrogram=False, normalize_signal=False, eps=eps, mode='melbanks', low_freq=f_min, high_freq=f_max, num_bins=n_mels, log_mels=False, fft_mode='abs', sqrt_real_imag=False, return_img=False))
        self.eps = eps
        if do_spec_aug:
            self.specaug = FbankAug(freq_start_bin=freq_start_bin, freq_mask_width=freq_mask_width, time_mask_width=time_mask_width)
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.amp.autocast(enabled=False):
                x = self.torchfbank(x) + self.eps
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    x = self.specaug(x)
        return x


class TFSpectrogram(nn.Module):

    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600, n_mels=80, window='hamming', normalize_spectrogram=False, normalize_signal=False, mode='fft', fft_mode='abs', pool_freqs=(2, 1), do_spec_aug=False, norm_signal=False, do_preemph=True, freq_start_bin=0, num_apply_spec_aug=1, freq_mask_width=(0, 8), time_mask_width=(0, 10), eps=1e-08):
        super(TFSpectrogram, self).__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.spectrogram = torch.nn.Sequential(NormalizeAudio() if norm_signal else nn.Identity(), PreEmphasis() if do_preemph else nn.Identity(), SpectralFeaturesTF(frame_length=win_length, frame_step=hop_length, fft_length=n_fft, sample_rate=sample_rate, window=window, eps=eps, mode=mode, low_freq=f_min, high_freq=f_max, num_bins=n_mels, normalize_spectrogram=False, normalize_signal=False, fft_mode='abs', log_mels=False, sqrt_real_imag=False, return_img=False))
        if pool_freqs is not None:
            self.pool_freq = nn.AvgPool2d(pool_freqs, stride=pool_freqs)
        else:
            self.pool_freq = nn.Identity()
        self.eps = eps
        if do_spec_aug:
            self.specaug = FbankAug(freq_start_bin=freq_start_bin, freq_mask_width=freq_mask_width, time_mask_width=time_mask_width)
        else:
            self.specaug = nn.Identity()

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.amp.autocast(enabled=False):
                x = self.spectrogram(x) + self.eps
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    for _ in range(self.num_apply_spec_aug):
                        x = self.specaug(x)
                x = self.pool_freq(x.unsqueeze(1))
        return x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, T, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, T).
    """

    def __init__(self, C, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.C = C,

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.C, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            w = self.weight
            b = self.bias
            for _ in range(x.ndim - 2):
                w = w.unsqueeze(-1)
                b = b.unsqueeze(-1)
            x = w * x + b
            return x

    def extra_repr(self) ->str:
        return ', '.join([f'{k}={v}' for k, v in {'C': self.C, 'data_format': self.data_format, 'eps': self.eps}.items()])


class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        if x.ndim == 3:
            pooling_mean = x.mean(dim=-1)
        elif x.ndim == 4:
            pooling_mean = x.mean(dim=(-1, -2))
        pooling_mean = pooling_mean.flatten(start_dim=1)
        return pooling_mean

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TMAP(nn.Module):
    """
    Temporal max-average pooling, only first-order mean is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TMAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        if x.ndim == 3:
            x1, _ = torch.max(x, dim=-1)
            x2 = torch.mean(x, dim=-1)
        elif x.ndim == 4:
            x1 = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
            x2 = torch.mean(x, dim=(-1, -2))
        max_avg = x1 + x2
        return max_avg

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSDP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-07)
        pooling_std = pooling_std.flatten(start_dim=1)
        return pooling_std

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-07)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


class TSTP_var(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pooling_mean = x.mean(dim=-1)
        pooling_var = torch.var(x, dim=-1)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_var = pooling_var.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_var), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False, **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-07).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-07))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MHASTP(torch.nn.Module):
    """ Multi head attentive statistics pooling
    Reference:
        Self Multi-Head Attention for Speaker Recognition
        https://arxiv.org/pdf/1906.09890.pdf
    """

    def __init__(self, in_dim, layer_num=2, head_num=2, d_s=1, bottleneck_dim=64, **kwargs):
        super(MHASTP, self).__init__()
        assert in_dim % head_num == 0
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        heads_att_trans = []
        for i in range(self.head_num):
            att_trans = nn.Sequential()
            for i in range(layer_num - 1):
                att_trans.add_module('att_' + str(i), nn.Conv1d(channel_dims[i], channel_dims[i + 1], 1, 1))
                att_trans.add_module('tanh' + str(i), nn.Tanh())
            att_trans.add_module('att_' + str(layer_num - 1), nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num], 1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:
            input = input.reshape(input.shape[0], input.shape[1] * input.shape[2], input.shape[3])
        assert len(input.shape) == 3
        bs, f_dim, t_dim = input.shape
        chunks = torch.chunk(input, self.head_num, 1)
        chunks_out = []
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i])
            alpha = F.softmax(att_score, dim=-1)
            mean = torch.sum(alpha * chunks[i], dim=2)
            var = torch.sum(alpha * chunks[i] ** 2, dim=2) - mean ** 2
            std = torch.sqrt(var.clamp(min=1e-07))
            chunks_out.append(torch.cat((mean, std), dim=1))
        out = torch.cat(chunks_out, dim=1)
        return out

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MQMHASTP(torch.nn.Module):
    """ An attentive pooling
    Reference:
        multi query multi head attentive statistics pooling
        https://arxiv.org/pdf/2110.05042.pdf
    Args:
        in_dim: the feature dimension of input
        layer_num: the number of layer in the pooling layer
        query_num: the number of querys
        head_num: the number of heads
        bottleneck_dim: the bottleneck dimension

    SA (H = 1, Q = 1, n = 2, d_s = 1) ref:
        https://www.danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    MHA (H > 1, Q = 1, n = 1, d_s = 1) ref:
        https://arxiv.org/pdf/1906.09890.pdf
    AS (H = 1, Q > 1, n = 2, d_s = 1) ref:
        https://arxiv.org/pdf/1803.10963.pdf
    VSA (H = 1, Q > 1, n = 2, d_s = d_h) ref:
        http://www.interspeech2020.org/uploadfile/pdf/Mon-2-10-5.pdf
    """

    def __init__(self, in_dim, layer_num=2, query_num=2, head_num=8, d_s=2, bottleneck_dim=64, **kwargs):
        super(MQMHASTP, self).__init__()
        self.n_query = nn.ModuleList([MHASTP(in_dim, layer_num=layer_num, head_num=head_num, d_s=d_s, bottleneck_dim=bottleneck_dim) for i in range(query_num)])
        self.query_num = query_num
        self.in_dim = in_dim

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:
            input = input.reshape(input.shape[0], input.shape[1] * input.shape[2], input.shape[3])
        assert len(input.shape) == 3
        res = []
        for i, layer in enumerate(self.n_query):
            res.append(layer(input))
        out = torch.cat(res, dim=-1)
        return out

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2 * self.query_num
        return self.out_dim


class to1d(nn.Module):

    def forward(self, x):
        size = x.size()
        bs, c, f, t = size
        return x.permute((0, 2, 1, 3)).reshape((bs, c * f, t))


class to2d(nn.Module):

    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self, x):
        size = x.size()
        bs, cf, t = size
        out = x.reshape((bs, self.f, self.c, t)).permute((0, 2, 1, 3))
        return out

    def extra_repr(self) ->str:
        return f'f={self.f},c={self.c}'


class to1d_tfopt(nn.Module):

    def forward(self, x):
        bs, c, t, f = x.size()
        return x.permute((0, 3, 1, 2)).reshape((bs, c * f, t))


class to2d_tfopt(nn.Module):

    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self, x):
        bs, cf, t = x.size()
        out = x.reshape((bs, self.f, self.c, t))
        out = out.permute((0, 2, 3, 1))
        return out

    def extra_repr(self) ->str:
        return f'f={self.f},c={self.c}'


class weigth1d(nn.Module):

    def __init__(self, N, C, sequential=False, requires_grad=True):
        super().__init__()
        self.N = N
        self.sequential = sequential
        self.w = nn.Parameter(torch.zeros(1, N, C, 1), requires_grad=requires_grad)

    def forward(self, xs):
        w = F.softmax(self.w, dim=1)
        if not self.sequential:
            xs = torch.cat([t.unsqueeze(1) for t in xs], dim=1)
            x = (w * xs).sum(dim=1)
        else:
            s = torch.zeros_like(xs[0])
            for i, t in enumerate(xs):
                s += t * w[:, i, :, :]
            x = s
        return x

    def extra_repr(self) ->str:
        return f'w={tuple(self.w.size())},sequential={self.sequential}'


class fwSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    link: https://arxiv.org/pdf/1709.01507.pdf
    PyTorch implementation
    """

    def __init__(self, num_freq, num_feats=64):
        super(fwSEBlock, self).__init__()
        self.squeeze = nn.Linear(num_freq, num_feats)
        self.exitation = nn.Linear(num_feats, num_freq)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = torch.mean(inputs, dim=[1, 3])
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)
        x = x[:, None, :, None]
        x = inputs * x
        return x


class ResBasicBlock(nn.Module):

    def __init__(self, inc, outc, num_freq, stride=1, se_channels=64, Gdiv=4, use_fwSE=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, inc if Gdiv is not None else outc, kernel_size=3, stride=stride, padding=1, bias=False, groups=inc // Gdiv if Gdiv is not None else 1)
        if Gdiv is not None:
            self.conv1pw = nn.Conv2d(inc, outc, 1)
        else:
            self.conv1pw = nn.Identity()
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False, groups=outc // Gdiv if Gdiv is not None else 1)
        if Gdiv is not None:
            self.conv2pw = nn.Conv2d(outc, outc, 1)
        else:
            self.conv2pw = nn.Identity()
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
        if use_fwSE:
            self.se = fwSEBlock(num_freq, se_channels)
        else:
            self.se = nn.Identity()
        if outc != inc:
            self.downsample = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(outc))
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1pw(self.conv1(x))
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2pw(self.conv2(out))
        out = self.bn2(out)
        out = self.se(out)
        out += self.downsample(residual)
        out = self.relu(out)
        return out


class ConvBlock2d(nn.Module):

    def __init__(self, c, f, block_type='convnext_like', Gdiv=1):
        super().__init__()
        if block_type == 'convnext_like':
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=[(3, 3)], Gdiv=Gdiv, padding='same', activation='gelu')
        elif block_type == 'convnext_like_relu':
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=[(3, 3)], Gdiv=Gdiv, padding='same', activation='relu')
        elif block_type == 'basic_resnet':
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64, max(c, 32)), Gdiv=Gdiv, use_fwSE=False)
        elif block_type == 'basic_resnet_fwse':
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64, max(c, 32)), Gdiv=Gdiv, use_fwSE=True)
        else:
            raise NotImplemented()

    def forward(self, x):
        return self.conv_block(x)


class PosEncConv(nn.Module):

    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(C, C, ks, padding=ks // 2, groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-06, data_format='channels_first')

    def forward(self, x):
        return x + self.norm(self.conv(x))


class TimeContextBlock1d(nn.Module):

    def __init__(self, C, hC, pos_ker_sz=59, block_type='att', red_dim_conv=None, exp_dim_conv=None):
        super().__init__()
        assert pos_ker_sz
        self.red_dim_conv = nn.Sequential(nn.Conv1d(C, hC, 1), LayerNorm(hC, eps=1e-06, data_format='channels_first'))
        if block_type == 'fc':
            self.tcm = nn.Sequential(nn.Conv1d(hC, hC * 2, 1), LayerNorm(hC * 2, eps=1e-06, data_format='channels_first'), nn.GELU(), nn.Conv1d(hC * 2, hC, 1))
        elif block_type == 'conv':
            self.tcm = nn.Sequential(*[ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7, 15, 31], Gdiv=1, padding='same') for i in range(4)])
        elif block_type == 'att':
            self.tcm = nn.Sequential(PosEncConv(hC, ks=pos_ker_sz, groups=hC), TransformerEncoderLayer(n_state=hC, n_mlp=hC * 2, n_head=4))
        elif block_type == 'conv+att':
            self.tcm = nn.Sequential(ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding='same'), ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding='same'), ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding='same'), ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding='same'), TransformerEncoderLayer(n_state=hC, n_mlp=hC, n_head=4))
        else:
            raise NotImplemented()
        self.exp_dim_conv = nn.Conv1d(hC, C, 1)

    def forward(self, x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x


class ReDimNet(nn.Module):

    def __init__(self, F=72, C=12, block_1d_type='att', block_2d_type='convnext_like', stages_setup=[(1, 2, 1, [(3, 3)], None), (2, 3, 1, [(3, 3)], None), (3, 4, 1, [(3, 3)], 8), (2, 5, 1, [(3, 3)], 8), (1, 5, 1, [(7, 1)], 8), (2, 3, 1, [(3, 3)], 8)], group_divisor=1, out_channels=512, feat_agg_dropout=0.0, return_2d_output=False, return_all_outputs=False, offset_fm_weights=0, is_subnet=False):
        super().__init__()
        self.F = F
        self.C = C
        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type
        self.stages_setup = stages_setup
        self.feat_agg_dropout = feat_agg_dropout
        self.return_2d_output = return_2d_output
        self.is_subnet = is_subnet
        self.offset_fm_weights = offset_fm_weights
        self.return_all_outputs = return_all_outputs
        self.build(F, C, stages_setup, group_divisor, out_channels, offset_fm_weights, is_subnet)

    def build(self, F, C, stages_setup, group_divisor, out_channels, offset_fm_weights, is_subnet):
        self.F = F
        self.C = C
        c = C
        f = F
        s = 1
        self.num_stages = len(stages_setup)
        if not is_subnet:
            self.stem = nn.Sequential(nn.Conv2d(1, int(c), kernel_size=3, stride=1, padding='same'), LayerNorm(int(c), eps=1e-06, data_format='channels_first'), to1d())
        else:
            self.stem = nn.Sequential(weigth1d(N=offset_fm_weights, C=F * C), to2d(f=F, c=C), nn.Conv2d(C, C, kernel_size=3, stride=1, padding='same'), LayerNorm(C, eps=1e-06, data_format='channels_first'), to1d())
        Block1d = functools.partial(TimeContextBlock1d, block_type=self.block_1d_type)
        Block2d = functools.partial(ConvBlock2d, block_type=self.block_2d_type)
        self.stages_cfs = []
        for stage_ind, (stride, num_blocks, conv_exp, kernel_sizes, att_block_red) in enumerate(stages_setup):
            assert stride in [1, 2, 3]
            num_feats_to_weight = offset_fm_weights + stage_ind + 1
            layers = [weigth1d(N=num_feats_to_weight, C=F * C if num_feats_to_weight > 1 else 1, requires_grad=num_feats_to_weight > 1), to2d(f=f, c=c), nn.Conv2d(int(c), int(stride * c * conv_exp), kernel_size=(stride, 1), stride=(stride, 1), padding=0, groups=1)]
            self.stages_cfs.append((c, f))
            c = stride * c
            assert f % stride == 0
            f = f // stride
            for block_ind in range(num_blocks):
                layers.append(Block2d(c=int(c * conv_exp), f=f, Gdiv=group_divisor))
            if conv_exp != 1:
                _group_divisor = group_divisor
                layers.append(nn.Sequential(nn.Conv2d(int(c * conv_exp), c, kernel_size=(3, 3), stride=1, padding='same', groups=c // _group_divisor if _group_divisor is not None else 1), nn.BatchNorm2d(c, eps=1e-06), nn.ReLU() if 'relu' in self.block_1d_type and 'relu' in self.block_2d_type else nn.GELU(), nn.Conv2d(c, c, 1)))
            layers.append(to1d())
            if att_block_red is not None:
                layers.append(Block1d(C * F, hC=C * F // att_block_red))
            setattr(self, f'stage{stage_ind}', nn.Sequential(*layers))
        num_feats_to_weight_fin = offset_fm_weights + len(stages_setup) + 1
        self.fin_wght1d = weigth1d(N=num_feats_to_weight_fin, C=F * C, requires_grad=num_feats_to_weight > 1)
        if out_channels is not None:
            self.mfa = nn.Sequential(nn.Conv1d(self.F * self.C, out_channels, kernel_size=1, padding='same'), nn.BatchNorm1d(out_channels, affine=True))
        else:
            self.mfa = nn.Identity()
        if self.return_2d_output:
            self.fin_to2d = to2d(f=f, c=c)
        else:
            self.fin_to2d = nn.Identity()

    def run_stage(self, prev_outs_1d, stage_ind):
        stage = getattr(self, f'stage{stage_ind}')
        x = stage(prev_outs_1d)
        return x

    def forward(self, inp):
        if not self.is_subnet:
            x = self.stem(inp)
            outputs_1d = [x]
        else:
            assert isinstance(inp, list)
            outputs_1d = list(inp)
            x = self.stem(inp)
            outputs_1d.append(x)
        for stage_ind in range(self.num_stages):
            outputs_1d.append(F.dropout(self.run_stage(outputs_1d, stage_ind), p=self.feat_agg_dropout, training=self.training))
        x = self.fin_wght1d(outputs_1d)
        outputs_1d.append(x)
        x = self.mfa(self.fin_to2d(x))
        if self.return_all_outputs:
            return x, outputs_1d
        else:
            return x


class ReDimNetWrap(nn.Module):

    def __init__(self, F=72, C=16, block_1d_type='att', block_2d_type='convnext_like', stages_setup=[(1, 2, 1, [(3, 3)], 12), (2, 2, 1, [(3, 3)], 12), (1, 3, 1, [(3, 3)], 12), (2, 4, 1, [(3, 3)], 8), (1, 4, 1, [(3, 3)], 8), (2, 4, 1, [(3, 3)], 4)], group_divisor=4, out_channels=None, embed_dim=192, num_classes=None, class_dropout=0.0, feat_agg_dropout=0.0, head_activation=None, hop_length=160, pooling_func='ASTP', feat_type='pt', global_context_att=False, emb_bn=False, spec_params=dict(do_spec_aug=False, freq_mask_width=(0, 6), time_mask_width=(0, 8)), return_all_outputs=False, tf_optimized_arch=False):
        super().__init__()
        self.return_all_outputs = return_all_outputs
        self.tf_optimized_arch = tf_optimized_arch
        if tf_optimized_arch:
            _ReDimNet = ReDimNetTFOpt
        else:
            _ReDimNet = ReDimNet
        self.backbone = _ReDimNet(F, C, block_1d_type, block_2d_type, stages_setup, group_divisor, out_channels, feat_agg_dropout=feat_agg_dropout, return_2d_output=False, return_all_outputs=return_all_outputs, offset_fm_weights=0, is_subnet=False)
        if feat_type in ['pt', 'pt_mel']:
            self.spec = features.MelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif feat_type in ['tf', 'tf_mel']:
            self.spec = features_tf.TFMelBanks(n_mels=F, hop_length=hop_length, **spec_params)
        elif feat_type == 'tf_spec':
            self.spec = features_tf.TFSpectrogram(**spec_params)
        if out_channels is None:
            out_channels = C * F
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=out_channels, global_context_att=global_context_att)
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = None
        if num_classes is not None:
            self.cls_head = nn.Sequential(nn.ReLU(inplace=False), nn.Dropout(p=class_dropout, inplace=False), nn.Linear(embed_dim, num_classes), eval(head_activation) if head_activation is not None else nn.Identity())
        else:
            self.cls_head = None

    def forward(self, x):
        x = self.spec(x)
        if self.tf_optimized_arch:
            x = x.permute(0, 2, 1)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if self.return_all_outputs:
            out, all_outs_1d = self.backbone(x)
        else:
            out = self.backbone(x)
        out = self.bn(self.pool(out))
        out = self.linear(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        if self.cls_head is not None:
            out = self.cls_head(out)
        if self.return_all_outputs:
            return out, all_outs_1d
        else:
            return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASTP,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     True),
    (ClippedGELUActivation,
     lambda: ([], {'min': 4, 'max': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock2d,
     lambda: ([], {'c': 4, 'f': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNeXtLikeBlock,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FastGELUActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FbankAug,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'hidden_size': 4, 'intermediate_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GELUActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogSpec,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MHASTP,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (NewGELUActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormalizeAudio,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreEmphasis,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (QuickGELUActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBasicBlock,
     lambda: ([], {'inc': 4, 'outc': 4, 'num_freq': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLUActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TAP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TMAP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TSDP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TSTP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TimeContextBlock1d,
     lambda: ([], {'C': 4, 'hC': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransformerEncoderLayer,
     lambda: ([], {'n_state': 4, 'n_mlp': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (fwSEBlock,
     lambda: ([], {'num_freq': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (to1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (to1d_tfopt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (weigth1d,
     lambda: ([], {'N': 4, 'C': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_IDRnD_redimnet(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

