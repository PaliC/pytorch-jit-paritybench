import sys
_module = sys.modules[__name__]
del sys
datasets = _module
gpt = _module
statistic = _module
vqgan = _module
demo = _module
hinge = _module
lpips = _module
models = _module
clip = _module
codec = _module
discriminator = _module
gpt = _module
quantizer = _module
vqgan = _module
tokenizer = _module
tokenizer = _module
train_gpt = _module
train_vqgan = _module
utils = _module

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


import matplotlib.pyplot as plt


import torch


import torch.nn.functional as F


from torchvision.utils import make_grid


import torch.nn as nn


from torchvision import models


from collections import namedtuple


from collections import OrderedDict


import numpy as np


from torch import nn


from functools import lru_cache


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data.distributed import DistributedSampler


import time


from torch.nn.parallel import DistributedDataParallel


import math


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


class vgg16(torch.nn.Module):

    def __init__(self):
        super(vgg16, self).__init__()
        vgg = models.vgg16(pretrained=False)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg.features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg.features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg.features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg.features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg.features[x])

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
        self.net = vgg16()
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        state = torch.load(LIPIPS_PATH, map_location='cpu')
        self.load_state_dict(state)
        None

    def forward(self, input, target):
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = F.normalize(outs0[kk]), F.normalize(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [lins[kk].model(diffs[kk]).mean([1, 2, 3]) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: 'torch.Tensor'):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor'):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: 'torch.Tensor'):
        return self.resblocks(x)


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: 'torch.Tensor'):
        x = self.conv1(x)
        B, W = x.shape[:2]
        x = x.reshape(B, W, -1)
        x = x.permute(0, 2, 1)
        cls_embed = self.class_embedding.view(1, 1, -1).expand(B, -1, -1)
        x = torch.cat([cls_embed, x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):

    def __init__(self, embed_dim=1024, img_size=224, vision_layers=[3, 4, 6, 3], vision_width=64, vision_patch_size=None, context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12):
        super().__init__()
        self.img_size = img_size
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = VisualTransformer(input_resolution=img_size, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, images):
        return self.visual(images.type(self.dtype))

    def encode_text(self, texts):
        x = self.token_embedding(texts).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, images, texts):
        if images is None:
            return self.encode_text(texts)
        elif texts is None:
            return self.encode_image(images)
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(texts)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return image_embeds, text_embeds, self.logit_scale.exp()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: 'torch.Tensor'):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Upsample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class Downsample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = 0, 1, 0, 1
        x = F.pad(x, pad, value=0)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, dropout):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(32, in_c), nn.SiLU(), nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1), nn.GroupNorm(32, out_c), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1))
        self.has_shortcut = in_c != out_c
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.block(x)
        if self.has_shortcut:
            x = self.shortcut(x)
        return x + h


class AttnBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_c)
        self.attn = nn.MultiheadAttention(in_c, num_heads=1, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)
        out, _ = self.attn(h, h, h, need_weights=False)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        out = x + out
        return out


class Encoder(nn.Module):

    def __init__(self, in_c=3, ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0, resolution=256, z_channels=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_c, ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        blocks = []
        for level in range(len(ch_mult)):
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))
            if level != len(ch_mult) - 1:
                blocks.append(Downsample(block_in))
                curr_res = curr_res // 2
        self.down = nn.Sequential(*blocks)
        self.mid = nn.Sequential(ResnetBlock(block_in, block_in, dropout=dropout), AttnBlock(block_in), ResnetBlock(block_in, block_in, dropout=dropout))
        self.final = nn.Sequential(nn.GroupNorm(32, block_in), nn.SiLU(), nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1), nn.Conv2d(z_channels, z_channels, kernel_size=1))

    def forward(self, x):
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        h = self.final(h)
        return h


class Decoder(nn.Module):

    def __init__(self, ch=128, out_ch=3, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0, resolution=256, z_channels=256):
        super().__init__()
        block_in = ch * ch_mult[len(ch_mult) - 1]
        self.quant_conv_in = nn.Conv2d(z_channels, z_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Sequential(ResnetBlock(block_in, block_in, dropout=dropout), AttnBlock(block_in), ResnetBlock(block_in, block_in, dropout=dropout))
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out
            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2
        self.up = nn.Sequential(*blocks)
        self.final = nn.Sequential(nn.GroupNorm(32, block_in), nn.SiLU(), nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1))

    def forward(self, z):
        h = self.quant_conv_in(z)
        h = self.conv_in(h)
        h = self.mid(h)
        h = self.up(h)
        h = self.final(h)
        return h


class Discriminator(nn.Module):

    def __init__(self, in_c=3, ch=64, n_layer=3):
        super().__init__()
        modules = [nn.Conv2d(in_c, ch, kernel_size=4, stride=2, padding=1)]
        modules += [nn.LeakyReLU(0.2, True)]
        chs = [(ch * min(2 ** i, 8)) for i in range(n_layer + 1)]
        for i in range(1, n_layer + 1):
            stride = 2 if i != n_layer else 1
            modules += [nn.Conv2d(chs[i - 1], chs[i], kernel_size=4, stride=stride, padding=1, bias=False), nn.BatchNorm2d(chs[i]), nn.LeakyReLU(0.2, True)]
        self.features = nn.Sequential(*modules)
        self.head = nn.Conv2d(chs[-1], 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.features(x)
        out = self.head(x)
        return out


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout, block_size) ->None:
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(n_embd, n_head, n_embd * 4, dropout, activation=F.gelu, batch_first=True, norm_first=True)
        mask = torch.ones(block_size, block_size, dtype=torch.bool)
        mask = ~torch.tril(mask)
        self.register_buffer('mask', mask)

    def forward(self, x):
        L = x.size(1)
        assert L <= self.mask.size(0)
        mask = self.mask[:L, :L]
        return self.encoder(x, mask)


class GPT(nn.Module):

    def __init__(self, vocab_size, n_layer, n_embed, n_head, block_size=256, n_cond_embed=512, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embed))
        self.cond_proj = nn.Linear(n_cond_embed, n_embed)
        self.drop = nn.Dropout(dropout)
        blocks = [Block(n_embed, n_head, dropout, block_size) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size

    def forward(self, idx, embed):
        x = self.tok_emb(idx)
        embed = self.cond_proj(embed).unsqueeze(1)
        x = torch.cat([embed, x], dim=1)
        assert x.size(1) <= self.block_size
        x = x + self.pos_emb[:, :x.size(1)]
        x = self.drop(x)
        x = self.blocks(x)
        logits = self.head(self.norm(x))
        return logits

    @torch.no_grad()
    def sample(self, embed, steps, temperature=1.0, top_k=None, top_p=1.0):
        N = embed.size(0)
        indices = torch.zeros(N, 0).long()
        for _ in range(steps):
            logits = self(indices, embed)
            logits = logits[:, -1] / temperature
            logits = self.top_k_top_p(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            indices = torch.cat((indices, idx), dim=1)
        return indices

    @staticmethod
    def top_k_top_p(logits, top_k=None, top_p=1.0):
        if top_k is not None:
            assert 1 <= top_k <= logits.size(-1)
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[..., [-1]]] = -torch.inf
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)
            mask = cum_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0
            mask = mask.scatter(1, sorted_indices, mask)
            logits[mask] = -torch.inf
        return logits


class VectorQuantizer(nn.Module):

    def __init__(self, codebook_size, embed_dim, beta=0.2):
        super().__init__()
        self.beta = beta
        self.K = codebook_size
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, x):
        indices = self.encode(x)
        x_q = self.decode(indices)
        x_q = x_q.permute(0, 3, 1, 2)
        loss = F.mse_loss(x, x_q.detach()) + self.beta * F.mse_loss(x.detach(), x_q)
        x_q = x + (x_q - x).detach()
        return x_q, loss, indices

    def encode(self, x):
        B, C, H, W = x.shape
        vectors = x.permute(0, 2, 3, 1).reshape(-1, C)
        dist = torch.cdist(vectors, self.embedding.weight)
        indices = dist.argmin(1).view(B, H, W)
        return indices

    def decode(self, indices):
        embeddings = self.embedding(indices)
        return embeddings


class VQGAN(nn.Module):

    def __init__(self, codebook_size, n_embed):
        super().__init__()
        self.encoder = Encoder(z_channels=n_embed)
        self.decoder = Decoder(z_channels=n_embed)
        self.quantizer = VectorQuantizer(codebook_size, n_embed)
        self.discriminator = Discriminator()

    def encode(self, x):
        z = self.encoder(x)
        z_q, loss_q, indices = self.quantizer(z)
        return z_q, loss_q, indices

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x, stage: 'int'):
        if stage == 0:
            z, loss_q, _ = self.encode(x)
            x_recon = self.decode(z)
            logits_fake = self.discriminator(x_recon)
            return x_recon, loss_q, logits_fake
        elif stage == 1:
            with torch.no_grad():
                z, loss_q, _ = self.encode(x)
                x_recon = self.decode(z)
            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(x_recon.detach())
            return logits_real, logits_fake
        else:
            raise ValueError(f'Invalid stage: {stage}')


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'n_embd': 4, 'n_head': 4, 'dropout': 0.5, 'block_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     False),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Downsample,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VQGAN,
     lambda: ([], {'codebook_size': 4, 'n_embed': 4}),
     lambda: ([torch.rand([4, 3, 64, 64]), 0], {}),
     False),
    (VectorQuantizer,
     lambda: ([], {'codebook_size': 4, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_HFAiLab_clip_gen(_paritybench_base):
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

