import sys
_module = sys.modules[__name__]
del sys
arguments = _module
dataset = _module
eval_fsq = _module
lpips = _module
metric = _module
model = _module
quantizers = _module
fsq = _module
lfq = _module
vq = _module
scheduler = _module
train = _module
util = _module

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


from torchvision import datasets


from torchvision import transforms


from torchvision.utils import save_image


from torchvision.utils import make_grid


import torch.nn as nn


from torchvision import models


from collections import namedtuple


from torch import nn


from torch.nn import functional as F


import numpy as np


from typing import List


from typing import Optional


from torch.nn import Module


from torch import Tensor


from torch import int32


from math import log2


from math import ceil


from torch import einsum


import torch.nn.functional as F


from torch import distributed as dist


from torch.optim.lr_scheduler import _LRScheduler


import math


import random


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


CKPT_MAP = {'vgg_lpips': 'vgg.pth'}


MD5_MAP = {'vgg_lpips': 'd507d7349b931f0638a25a48a722f98a'}


URL_MAP = {'vgg_lpips': 'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1'}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            with open(local_path, 'wb') as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or check and not md5_hash(path) == MD5_MAP[name]:
        None
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
    return x / norm_factor


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):

    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name='vgg_lpips'):
        ckpt = get_ckpt_path(name, 'loss/pretrained/lpips')
        self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
        None

    @classmethod
    def from_pretrained(cls, name='vgg_lpips'):
        if name is not 'vgg_lpips':
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class Encoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        in_channel = args.in_channel
        channel = args.channel
        embed_dim = args.embed_dim
        blocks = [nn.Conv2d(in_channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 4, stride=2, padding=1)]
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        in_channel = args.embed_dim
        out_channel = args.in_channel
        channel = args.channel
        blocks = [nn.ConvTranspose2d(in_channel, channel, 4, stride=2, padding=1)]
        blocks.append(nn.ReLU(inplace=True))
        blocks.extend([nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channel, out_channel, 1)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def round_ste(z: 'Tensor') ->Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class FSQ(Module):

    def __init__(self, levels: 'List[int]', dim: 'Optional[int]'=None, num_codebooks=1, keep_num_codebooks_dim: 'Optional[bool]'=None, scale: 'Optional[float]'=None):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer('_levels', _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer('_basis', _basis, persistent=False)
        self.scale = scale
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.dim = default(dim, len(_levels) * num_codebooks)
        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections
        self.codebook_size = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer('implicit_codebook', implicit_codebook, persistent=False)

    def bound(self, z: 'Tensor', eps: 'float'=0.001) ->Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: 'Tensor') ->Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: 'Tensor') ->Tensor:
        half_width = self._levels // 2
        return zhat_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, zhat: 'Tensor') ->Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: 'Tensor') ->Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices: 'Tensor', project_out=True) ->Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= 3 + int(self.keep_num_codebooks_dim)
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = indices // self._basis % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')
        if project_out:
            codes = self.project_out(codes)
        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')
        return codes

    def forward(self, z: 'Tensor') ->Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        is_img_or_video = z.ndim >= 4
        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')
        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = rearrange(codes, 'b n c d -> b n (c d)')
        out = self.project_out(codes)
        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = unpack_one(indices, ps, 'b * c')
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')
        return out, indices


LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])


Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])


def log(t, eps=1e-05):
    return t.clamp(min=eps).log()


def entropy(prob):
    return -prob * log(prob)


def euclidean_distance_squared(x, y):
    x2 = reduce(x ** 2, '... n d -> ... n', 'sum')
    y2 = reduce(y ** 2, 'n d -> n', 'sum')
    xy = einsum('... i d, j d -> ... i j', x, y) * -2
    return rearrange(x2, '... i -> ... i 1') + y2 + xy


class LFQ(Module):

    def __init__(self, *, dim=None, codebook_size=None, entropy_loss_weight=0.1, commitment_loss_weight=1.0, diversity_gamma=2.5, straight_through_activation=nn.Identity(), num_codebooks=1, keep_num_codebooks_dim=None, codebook_scale=1.0):
        super().__init__()
        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'
        codebook_size = default(codebook_size, lambda : 2 ** dim)
        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        has_projections = dim != codebook_dims
        self.project_in = nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
        self.has_projections = has_projections
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.activation = straight_through_activation
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.codebook_scale = codebook_scale
        self.commitment_loss_weight = commitment_loss_weight
        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.0), persistent=False)
        all_codes = torch.arange(codebook_size)
        bits = (all_codes[..., None].int() & self.mask != 0).float()
        codebook = self.bits_to_codes(bits)
        self.register_buffer('codebook', codebook, persistent=False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= 3 + int(self.keep_num_codebooks_dim)
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')
        bits = indices[..., None].int() & self.mask != 0
        codes = self.bits_to_codes(bits)
        codes = rearrange(codes, '... c d -> ... (c d)')
        if project_out:
            codes = self.project_out(codes)
        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')
        return codes

    def forward(self, x, inv_temperature=1.0, return_loss_breakdown=False):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        is_img_or_video = x.ndim >= 4
        if is_img_or_video:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')
        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'
        x = self.project_in(x)
        x = rearrange(x, 'b n (c d) -> b n c d', c=self.num_codebooks)
        original_input = x
        codebook_value = torch.ones_like(x) * self.codebook_scale
        quantized = torch.where(x > 0, codebook_value, -codebook_value)
        if self.training:
            x = self.activation(x)
            x = x - x.detach() + quantized
        else:
            x = quantized
        indices = reduce((x > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')
        if self.training and self.entropy_loss_weight != 0:
            None
            exit()
            distance = euclidean_distance_squared(original_input, self.codebook)
            prob = (-distance * inv_temperature).softmax(dim=-1)
            per_sample_entropy = entropy(prob).mean()
            avg_prob = reduce(prob, 'b n c d -> b c d', 'mean')
            codebook_entropy = entropy(avg_prob).mean()
            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        else:
            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero
        if self.training:
            commit_loss = F.mse_loss(original_input, quantized.detach())
        else:
            commit_loss = self.zero
        x = rearrange(x, 'b n c d -> b n (c d)')
        x = self.project_out(x)
        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')
            indices = unpack_one(indices, ps, 'b * c')
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')
        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight
        ret = Return(x, indices, aux_loss)
        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


class VectorQuantizeEMA(nn.Module):

    def __init__(self, args, embedding_dim, n_embed, commitment_cost=1, decay=0.99, eps=1e-05):
        super().__init__()
        self.args = args
        self.ema = True if args.quantizer == 'ema' else False
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.embed.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', self.embed.weight.data.clone())
        self.decay = decay
        self.eps = eps

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_e = z_e.permute(0, 2, 3, 1)
        flatten = z_e.reshape(-1, self.embedding_dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed.weight.t() + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(B, H, W)
        z_q = self.embed_code(embed_ind)
        if self.training and self.ema:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = (flatten.transpose(0, 1) @ embed_onehot).transpose(0, 1)
            all_reduce(embed_onehot_sum.contiguous())
            all_reduce(embed_sum.contiguous())
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.weight.data.copy_(embed_normalized)
        if self.ema:
            diff = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean()
        else:
            diff = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class VQVAE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.quantizer == 'ema' or args.quantizer == 'origin':
            self.quantize_t = VectorQuantizeEMA(args, args.embed_dim, args.n_embed)
        elif args.quantizer == 'lfq':
            self.quantize_t = LFQ(codebook_size=2 ** args.lfq_dim, dim=args.lfq_dim, entropy_loss_weight=args.entropy_loss_weight, commitment_loss_weight=args.codebook_loss_weight)
        elif args.quantizer == 'fsq':
            self.quantize_t = FSQ(levels=args.levels)
        else:
            None
            exit()
        self.enc = Encoder(args)
        self.dec = Decoder(args)

    def forward(self, input, return_id=True):
        quant_t, diff, id_t = self.encode(input)
        dec = self.dec(quant_t)
        if return_id:
            return dec, diff, id_t
        return dec, diff

    def encode(self, input):
        logits = self.enc(input)
        if self.args.quantizer == 'ema' or self.args.quantizer == 'origin':
            quant_t, diff_t, id_t = self.quantize_t(logits)
            diff_t = diff_t.unsqueeze(0)
        elif self.args.quantizer == 'fsq':
            quant_t, id_t = self.quantize_t(logits)
            diff_t = torch.tensor(0.0).float()
        elif self.args.quantizer == 'lfq':
            quant_t, id_t, diff_t = self.quantize_t(logits)
        return quant_t, diff_t, id_t

    def decode(self, code):
        return self.dec(code)

    def decode_code(self, code_t):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec = self.dec(quant_t)
        return dec


class Quantize(nn.Module):

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-05):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        torch.nn.init.xavier_uniform_(embed, gain=torch.nn.init.calculate_gain('tanh'))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input, continuous_relax=False, temperature=1.0, hard=False):
        input = input.permute(0, 2, 3, 1)
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        if not continuous_relax:
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        elif not hard:
            embed_soft, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=False)
            embed_ind = embed_ind.view(*input.shape[:-1])
            embed_soft = embed_soft.view(*input.shape[:-1], self.n_embed)
            quantize = embed_soft @ self.embed.transpose(0, 1)
        else:
            embed_onehot, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=True)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        if self.training and (continuous_relax and hard or not continuous_relax):
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        if not continuous_relax:
            diff = (quantize.detach() - input).pow(2).mean()
            quantize = input + (quantize - input).detach()
        else:
            qy = (-dist).softmax(-1)
            diff = torch.sum(qy * torch.log(qy * self.n_embed + 1e-20), dim=-1).mean()
            quantize = quantize
        quantize = quantize.permute(0, 3, 1, 2)
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

