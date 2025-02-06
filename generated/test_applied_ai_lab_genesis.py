import sys
_module = sys.modules[__name__]
del sys
apc_config = _module
gqn_config = _module
multi_object_config = _module
multid_config = _module
shapestacks_config = _module
sketchy_config = _module
models = _module
genesis_config = _module
genesisv2_config = _module
monet_config = _module
vae_config = _module
modules = _module
attention = _module
blocks = _module
component_vae = _module
decoders = _module
encoders = _module
unet = _module
scripts = _module
compute_fid = _module
compute_seg_metrics = _module
generate_multid = _module
sketchy_preparation = _module
visualise_data = _module
visualise_generation = _module
visualise_reconstruction = _module
clevr_with_masks = _module
multi_dsprites = _module
objects_room = _module
segmentation_metrics = _module
tetrominoes = _module
fid_score = _module
inception = _module
shapestacks = _module
segmentation_utils = _module
shapestacks_provider = _module
VAE = _module
sylvester = _module
layers = _module
tf_gqn = _module
gqn_tfr_provider = _module
train = _module
utils = _module
geco = _module
misc = _module
plotting = _module

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


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


import torch.nn.functional as F


import tensorflow as tf


import numpy as np


import torch.nn as nn


from torch.distributions.normal import Normal


from torch.distributions.categorical import Categorical


from torch.distributions.kl import kl_divergence


from torch.nn import Sequential as Seq


from random import randint


from random import choice


import matplotlib.pyplot as plt


from matplotlib.colors import NoNorm


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


from torchvision import models


import time


import torch.optim as optim


from torchvision.utils import make_grid


from sklearn.metrics import adjusted_rand_score


class BroadcastDecoder(nn.Module):

    def __init__(self, in_chnls, out_chnls, h_chnls, num_layers, img_dim, act):
        super(BroadcastDecoder, self).__init__()
        broad_dim = img_dim + 2 * num_layers
        mods = [B.BroadcastLayer(broad_dim), nn.Conv2d(in_chnls + 2, h_chnls, 3), act]
        for _ in range(num_layers - 1):
            mods.extend([nn.Conv2d(h_chnls, h_chnls, 3), act])
        mods.append(nn.Conv2d(h_chnls, out_chnls, 1))
        self.seq = nn.Sequential(*mods)

    def forward(self, x):
        return self.seq(x)


class MONetCompEncoder(nn.Module):

    def __init__(self, cfg, act):
        super(MONetCompEncoder, self).__init__()
        nin = cfg.input_channels if hasattr(cfg, 'input_channels') else 3
        c = cfg.comp_enc_channels
        self.ldim = cfg.comp_ldim
        nin_mlp = 2 * c * (cfg.img_size // 16) ** 2
        nhid_mlp = max(256, 2 * self.ldim)
        self.module = Seq(nn.Conv2d(nin + 1, c, 3, 2, 1), act, nn.Conv2d(c, c, 3, 2, 1), act, nn.Conv2d(c, 2 * c, 3, 2, 1), act, nn.Conv2d(2 * c, 2 * c, 3, 2, 1), act, B.Flatten(), nn.Linear(nin_mlp, nhid_mlp), act, nn.Linear(nhid_mlp, 2 * self.ldim))

    def forward(self, x):
        return self.module(x)


class ComponentVAE(nn.Module):

    def __init__(self, nout, cfg, act):
        super(ComponentVAE, self).__init__()
        self.ldim = cfg.comp_ldim
        self.montecarlo = cfg.montecarlo_kl
        self.pixel_bound = cfg.pixel_bound
        self.encoder_module = MONetCompEncoder(cfg=cfg, act=act)
        self.decoder_module = BroadcastDecoder(in_chnls=self.ldim, out_chnls=nout, h_chnls=cfg.comp_dec_channels, num_layers=cfg.comp_dec_layers, img_dim=cfg.img_size, act=act)

    def forward(self, x, log_mask):
        """
        Args:
            x (torch.Tensor): Input to reconstruct [batch size, 3, dim, dim]
            log_mask (torch.Tensor or list of torch.Tensors):
                Mask to reconstruct [batch size, 1, dim, dim]
        """
        K = 1
        b_sz = x.size(0)
        if isinstance(log_mask, list) or isinstance(log_mask, tuple):
            K = len(log_mask)
            x = x.repeat(K, 1, 1, 1)
            log_mask = torch.cat(log_mask, dim=0)
        x = torch.cat((log_mask, x), dim=1)
        mu, sigma = self.encode(x)
        q_z = Normal(mu, sigma)
        z = q_z.rsample()
        x_r = self.decode(z)
        x_r_k = torch.chunk(x_r, K, dim=0)
        z_k = torch.chunk(z, K, dim=0)
        mu_k = torch.chunk(mu, K, dim=0)
        sigma_k = torch.chunk(sigma, K, dim=0)
        stats = AttrDict(mu_k=mu_k, sigma_k=sigma_k, z_k=z_k)
        return x_r_k, stats

    def encode(self, x):
        x = self.encoder_module(x)
        mu, sigma_ps = torch.chunk(x, 2, dim=1)
        sigma = B.to_sigma(sigma_ps)
        return mu, sigma

    def decode(self, z):
        x_hat = self.decoder_module(z)
        if self.pixel_bound:
            x_hat = torch.sigmoid(x_hat)
        return x_hat

    def sample(self, batch_size=1, steps=1):
        raise NotImplementedError


class Genesis(nn.Module):

    def __init__(self, cfg):
        super(Genesis, self).__init__()
        self.K_steps = cfg.K_steps
        self.img_size = cfg.img_size
        self.two_stage = cfg.two_stage
        self.autoreg_prior = cfg.autoreg_prior
        self.comp_prior = False
        if self.two_stage and self.K_steps > 1:
            self.comp_prior = cfg.comp_prior
        self.ldim = cfg.attention_latents
        self.pixel_bound = cfg.pixel_bound
        if not hasattr(cfg, 'comp_symmetric'):
            cfg.comp_symmetric = False
        self.debug = cfg.debug
        assert cfg.montecarlo_kl == True
        if hasattr(cfg, 'input_channels'):
            input_channels = cfg.input_channels
        else:
            input_channels = 3
        att_nin = input_channels
        att_nout = 1
        att_core = sylvester.VAE(self.ldim, [att_nin, cfg.img_size, cfg.img_size], att_nout, cfg.enc_norm, cfg.dec_norm)
        if self.K_steps > 1:
            self.att_steps = self.K_steps
            self.att_process = attention.LatentSBP(att_core)
        if self.two_stage:
            self.comp_vae = ComponentVAE(nout=input_channels, cfg=cfg, act=nn.ELU())
            if cfg.comp_symmetric:
                self.comp_vae.encoder_module = nn.Sequential(sylvester.build_gc_encoder([input_channels + 1, 32, 32, 64, 64], [32, 32, 64, 64, 64], [1, 2, 1, 2, 1], 2 * cfg.comp_ldim, att_core.last_kernel_size, hn=cfg.enc_norm, gn=cfg.enc_norm), B.Flatten())
                self.comp_vae.decoder_module = nn.Sequential(B.UnFlatten(), sylvester.build_gc_decoder([64, 64, 32, 32, 32], [64, 32, 32, 32, 32], [1, 2, 1, 2, 1], cfg.comp_ldim, att_core.last_kernel_size, hn=cfg.dec_norm, gn=cfg.dec_norm), nn.Conv2d(32, input_channels, 1))
        else:
            assert self.K_steps > 1
            self.decoder = decoders.BroadcastDecoder(in_chnls=self.ldim, out_chnls=input_channels, h_chnls=cfg.comp_dec_channels, num_layers=cfg.comp_dec_layers, img_dim=self.img_size, act=nn.ELU())
        if self.autoreg_prior and self.K_steps > 1:
            self.prior_lstm = nn.LSTM(self.ldim, 256)
            self.prior_linear = nn.Linear(256, 2 * self.ldim)
        if self.comp_prior and self.two_stage and self.K_steps > 1:
            self.prior_mlp = nn.Sequential(nn.Linear(self.ldim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 2 * cfg.comp_ldim))
        std = cfg.pixel_std2 * torch.ones(1, 1, 1, 1, self.K_steps)
        std[0, 0, 0, 0, 0] = cfg.pixel_std1
        self.register_buffer('std', std)

    def forward(self, x):
        """
        Performs a forward pass in the model.

        Args:
          x (torch.Tensor): input images [batch size, 3, dim, dim]

        Returns:
          recon: reconstructed images [N, 3, H, W]
          losses: 
          stats: 
          att_stats: 
          comp_stats: 
        """
        if self.K_steps > 1:
            log_m_k, log_s_k, att_stats = self.att_process(x, self.att_steps)
        else:
            log_m_k = [torch.zeros_like(x[:, :1, :, :])]
            log_s_k = [-10000000000.0 * torch.ones_like(x[:, :1, :, :])]
            att_stats = None
        if len(log_m_k) == self.K_steps + 1:
            del log_m_k[-1]
            log_m_k[self.K_steps - 1] = log_s_k[self.K_steps - 1]
        if self.debug or not self.training:
            assert len(log_m_k) == self.K_steps
        if self.two_stage:
            x_r_k, comp_stats = self.comp_vae(x, log_m_k)
        else:
            z_batched = torch.cat(att_stats.z_k, dim=0)
            x_r_batched = self.decoder(z_batched)
            x_r_k = torch.chunk(x_r_batched, self.K_steps, dim=0)
            if self.pixel_bound:
                x_r_k = [torch.sigmoid(x) for x in x_r_k]
            comp_stats = None
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        recon = (m_stack * x_r_stack).sum(dim=4)
        losses = AttrDict()
        losses['err'] = self.x_loss(x, log_m_k, x_r_k, self.std)
        if self.K_steps > 1:
            if 'zm_0_k' in att_stats and 'zm_k_k' in att_stats:
                q_zm_0_k = [Normal(m, s) for m, s in zip(att_stats.mu_k, att_stats.sigma_k)]
                zm_0_k = att_stats.z_0_k
                zm_k_k = att_stats.z_k_k
                ldj_k = att_stats.ldj_k
            elif 'mu_k' in att_stats and 'sigma_k' in att_stats:
                q_zm_0_k = [Normal(m, s) for m, s in zip(att_stats.mu_k, att_stats.sigma_k)]
                zm_0_k = att_stats.z_k
                zm_k_k = att_stats.z_k
                ldj_k = None
            losses['kl_m_k'], p_zm_k = self.mask_latent_loss(q_zm_0_k, zm_0_k, zm_k_k, ldj_k, self.prior_lstm, self.prior_linear, debug=self.debug or not self.training)
            att_stats['pmu_k'] = [p_zm.mean for p_zm in p_zm_k]
            att_stats['psigma_k'] = [p_zm.scale for p_zm in p_zm_k]
            if self.debug or not self.training:
                assert len(zm_k_k) == self.K_steps
                assert len(zm_0_k) == self.K_steps
                if ldj_k is not None:
                    assert len(ldj_k) == self.K_steps
        else:
            losses['kl_m'] = torch.tensor(0.0)
        if self.two_stage:
            losses['kl_l_k'] = []
            if self.comp_prior:
                comp_stats['pmu_k'], comp_stats['psigma_k'] = [], []
                for step, zl in enumerate(comp_stats.z_k):
                    mlp_out = self.prior_mlp(zm_k_k[step])
                    mlp_out = torch.chunk(mlp_out, 2, dim=1)
                    mu = torch.tanh(mlp_out[0])
                    sigma = B.to_prior_sigma(mlp_out[1])
                    p_zl = Normal(mu, sigma)
                    comp_stats['pmu_k'].append(mu)
                    comp_stats['psigma_k'].append(sigma)
                    q_zl = Normal(comp_stats.mu_k[step], comp_stats.sigma_k[step])
                    kld = (q_zl.log_prob(zl) - p_zl.log_prob(zl)).sum(dim=1)
                    losses['kl_l_k'].append(kld)
                if self.debug or not self.training:
                    assert len(comp_stats['pmu_k']) == self.K_steps
                    assert len(comp_stats['psigma_k']) == self.K_steps
            else:
                p_zl = Normal(0, 1)
                for step, zl in enumerate(comp_stats.z_k):
                    q_zl = Normal(comp_stats.mu_k[step], comp_stats.sigma_k[step])
                    kld = (q_zl.log_prob(zl) - p_zl.log_prob(zl)).sum(dim=1)
                    losses['kl_l_k'].append(kld)
            if self.debug or not self.training:
                assert len(comp_stats.z_k) == self.K_steps
                assert len(losses['kl_l_k']) == self.K_steps
        stats = AttrDict(recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k, mx_r_k=[(x * logm.exp()) for x, logm in zip(x_r_k, log_m_k)])
        if self.debug or not self.training:
            assert len(log_m_k) == self.K_steps
            misc.check_log_masks(log_m_k)
        return recon, losses, stats, att_stats, comp_stats

    @staticmethod
    def x_loss(x, log_m_k, x_r_k, std, pixel_wise=False):
        p_xr_stack = Normal(torch.stack(x_r_k, dim=4), std)
        log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
        log_m_stack = torch.stack(log_m_k, dim=4)
        log_mx = log_m_stack + log_xr_stack
        err_ppc = -torch.log(log_mx.exp().sum(dim=4))
        if pixel_wise:
            return err_ppc
        else:
            return err_ppc.sum(dim=(1, 2, 3))

    @staticmethod
    def mask_latent_loss(q_zm_0_k, zm_0_k, zm_k_k=None, ldj_k=None, prior_lstm=None, prior_linear=None, debug=False):
        num_steps = len(zm_0_k)
        batch_size = zm_0_k[0].size(0)
        latent_dim = zm_0_k[0].size(1)
        if zm_k_k is None:
            zm_k_k = zm_0_k
        if prior_lstm is not None and prior_linear is not None:
            zm_seq = torch.cat([zm.view(1, batch_size, -1) for zm in zm_k_k[:-1]], dim=0)
            lstm_out, _ = prior_lstm(zm_seq)
            linear_out = prior_linear(lstm_out)
            linear_out = torch.chunk(linear_out, 2, dim=2)
            mu_raw = torch.tanh(linear_out[0])
            sigma_raw = B.to_prior_sigma(linear_out[1])
            mu_k = torch.split(mu_raw, 1, dim=0)
            sigma_k = torch.split(sigma_raw, 1, dim=0)
            p_zm_k = [Normal(0, 1)]
            for mean, std in zip(mu_k, sigma_k):
                p_zm_k += [Normal(mean.view(batch_size, latent_dim), std.view(batch_size, latent_dim))]
            if debug:
                assert zm_seq.size(0) == num_steps - 1
        else:
            p_zm_k = num_steps * [Normal(0, 1)]
        kl_m_k = []
        for step, p_zm in enumerate(p_zm_k):
            log_q = q_zm_0_k[step].log_prob(zm_0_k[step]).sum(dim=1)
            log_p = p_zm.log_prob(zm_k_k[step]).sum(dim=1)
            kld = log_q - log_p
            if ldj_k is not None:
                ldj = ldj_k[step].sum(dim=1)
                kld = kld - ldj
            kl_m_k.append(kld)
        if debug:
            assert len(p_zm_k) == num_steps
            assert len(kl_m_k) == num_steps
        return kl_m_k, p_zm_k

    def sample(self, batch_size, K_steps=None):
        if self.K_steps == 1:
            raise NotImplementedError
        K_steps = self.K_steps if K_steps is None else K_steps
        if self.autoreg_prior:
            zm_k = [Normal(0, 1).sample([batch_size, self.ldim])]
            state = None
            for k in range(1, self.att_steps):
                lstm_out, state = self.prior_lstm(zm_k[-1].view(1, batch_size, -1), state)
                linear_out = self.prior_linear(lstm_out)
                mu = linear_out[0, :, :self.ldim]
                sigma = B.to_prior_sigma(linear_out[0, :, self.ldim:])
                p_zm = Normal(mu.view([batch_size, self.ldim]), sigma.view([batch_size, self.ldim]))
                zm_k.append(p_zm.sample())
        else:
            p_zm = Normal(0, 1)
            zm_k = [p_zm.sample([batch_size, self.ldim]) for _ in range(self.att_steps)]
        log_m_k, log_s_k, out_k = self.att_process.masks_from_zm_k(zm_k, self.img_size)
        if len(log_m_k) == self.K_steps + 1:
            del log_m_k[-1]
            log_m_k[self.K_steps - 1] = log_s_k[self.K_steps - 1]
        assert len(zm_k) == self.K_steps
        assert len(log_m_k) == self.K_steps
        if self.two_stage:
            assert out_k[0].size(1) == 0
        else:
            assert out_k[0].size(1) == 0
        misc.check_log_masks(log_m_k)
        if self.two_stage:
            if self.comp_prior:
                zc_k = []
                for zm in zm_k:
                    mlp_out = torch.chunk(self.prior_mlp(zm), 2, dim=1)
                    mu = torch.tanh(mlp_out[0])
                    sigma = B.to_prior_sigma(mlp_out[1])
                    zc_k.append(Normal(mu, sigma).sample())
            else:
                zc_k = [Normal(0, 1).sample([batch_size, self.comp_vae.ldim]) for _ in range(K_steps)]
            zc_batch = torch.cat(zc_k, dim=0)
            x_batch = self.comp_vae.decode(zc_batch)
            x_k = list(torch.chunk(x_batch, self.K_steps, dim=0))
        else:
            zm_batched = torch.cat(zm_k, dim=0)
            x_batched = self.decoder(zm_batched)
            x_k = torch.chunk(x_batched, self.K_steps, dim=0)
            if self.pixel_bound:
                x_k = [torch.sigmoid(x) for x in x_k]
        assert len(x_k) == self.K_steps
        assert len(log_m_k) == self.K_steps
        if self.two_stage:
            assert len(zc_k) == self.K_steps
        x_stack = torch.stack(x_k, dim=4)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        generated_image = (m_stack * x_stack).sum(dim=4)
        stats = AttrDict(x_k=x_k, log_m_k=log_m_k, log_s_k=log_s_k, mx_k=[(x * m.exp()) for x, m in zip(x_k, log_m_k)])
        return generated_image, stats

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, att_stats, comp_stats = self.forward(image_batch)
        if self.two_stage:
            zm_k = att_stats['z_k'][:self.K_steps - 1]
            zc_k = comp_stats['z_k']
            return torch.cat([*zm_k, *zc_k], dim=1)
        else:
            zm_k = att_stats['z_k']
            return torch.cat(zm_k, dim=1)


class UNet(nn.Module):

    def __init__(self, num_blocks, img_size=64, filter_start=32, in_chnls=4, out_chnls=1, norm='in'):
        super(UNet, self).__init__()
        c = filter_start
        if norm == 'in':
            conv_block = B.ConvINReLU
        elif norm == 'gn':
            conv_block = B.ConvGNReLU
        else:
            conv_block = B.ConvReLU
        if num_blocks == 4:
            enc_in = [in_chnls, c, 2 * c, 2 * c]
            enc_out = [c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2 * c, 2 * c]
            enc_out = [c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2 * c, 2 * c]
            enc_out = [c, c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c, c]
        self.down = []
        self.up = []
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.featuremap_size = img_size // 2 ** (num_blocks - 1)
        self.mlp = nn.Sequential(B.Flatten(), nn.Linear(2 * c * self.featuremap_size ** 2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2 * c * self.featuremap_size ** 2), nn.ReLU())
        self.final_conv = nn.Conv2d(c, out_chnls, 1)
        self.out_chnls = out_chnls

    def forward(self, x):
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down) - 1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest')
            x_down.append(act)
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1, self.featuremap_size, self.featuremap_size)
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up) - 1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest')
        return self.final_conv(x_up), None


class MONet(nn.Module):

    def __init__(self, cfg):
        super(MONet, self).__init__()
        self.K_steps = cfg.K_steps
        self.prior_mode = cfg.prior_mode
        self.mckl = cfg.montecarlo_kl
        self.debug = cfg.debug
        self.pixel_bound = cfg.pixel_bound
        if not hasattr(cfg, 'filter_start'):
            cfg['filter_start'] = 32
        core = UNet(num_blocks=int(np.log2(cfg.img_size) - 1), img_size=cfg.img_size, filter_start=cfg.filter_start, in_chnls=4, out_chnls=1, norm='in')
        self.att_process = attention.SimpleSBP(core)
        self.comp_vae = ComponentVAE(nout=4, cfg=cfg, act=nn.ReLU())
        self.comp_vae.pixel_bound = False
        std = cfg.pixel_std2 * torch.ones(1, 1, 1, 1, self.K_steps)
        std[0, 0, 0, 0, 0] = cfg.pixel_std1
        self.register_buffer('std', std)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images [batch size, 3, dim, dim]
        """
        log_m_k, log_s_k, att_stats = self.att_process(x, self.K_steps - 1)
        x_m_r_k, comp_stats = self.comp_vae(x, log_m_k)
        x_r_k = [item[:, :3, :, :] for item in x_m_r_k]
        m_r_logits_k = [item[:, 3:, :, :] for item in x_m_r_k]
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        recon = (m_stack * x_r_stack).sum(dim=4)
        log_m_r_stack = self.get_mask_recon_stack(m_r_logits_k, self.prior_mode, log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
        losses = AttrDict()
        losses['err'] = Genesis.x_loss(x, log_m_k, x_r_k, self.std)
        losses['kl_m'] = self.kl_m_loss(log_m_k=log_m_k, log_m_r_k=log_m_r_k)
        q_z_k = [Normal(m, s) for m, s in zip(comp_stats.mu_k, comp_stats.sigma_k)]
        kl_l_k = misc.get_kl(comp_stats.z_k, q_z_k, len(q_z_k) * [Normal(0, 1)], self.mckl)
        losses['kl_l_k'] = [kld.sum(1) for kld in kl_l_k]
        stats = AttrDict(recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k, log_m_r_k=log_m_r_k, mx_r_k=[(x * logm.exp()) for x, logm in zip(x_r_k, log_m_k)])
        if self.debug:
            assert len(log_m_k) == self.K_steps
            assert len(log_m_r_k) == self.K_steps
            misc.check_log_masks(log_m_k)
            misc.check_log_masks(log_m_r_k)
        return recon, losses, stats, att_stats, comp_stats

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, _, comp_stats = self.forward(image_batch)
            return torch.cat(comp_stats.z_k, dim=1)

    @staticmethod
    def get_mask_recon_stack(m_r_logits_k, prior_mode, log):
        if prior_mode == 'softmax':
            if log:
                return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
            return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        elif prior_mode == 'scope':
            log_m_r_k = []
            log_s = torch.zeros_like(m_r_logits_k[0])
            for step, logits in enumerate(m_r_logits_k):
                if step == len(m_r_logits_k) - 1:
                    log_m_r_k.append(log_s)
                else:
                    log_a = F.logsigmoid(logits)
                    log_neg_a = F.logsigmoid(-logits)
                    log_m_r_k.append(log_s + log_a)
                    log_s = log_s + log_neg_a
            log_m_r_stack = torch.stack(log_m_r_k, dim=4)
            return log_m_r_stack if log else log_m_r_stack.exp()
        else:
            raise ValueError('No valid prior mode.')

    @staticmethod
    def kl_m_loss(log_m_k, log_m_r_k, debug=False):
        if debug:
            assert len(log_m_k) == len(log_m_r_k)
        batch_size = log_m_k[0].size(0)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        m_stack = torch.max(m_stack, torch.tensor(1e-05))
        m_r_stack = torch.max(m_r_stack, torch.tensor(1e-05))
        q_m = Categorical(m_stack.view(-1, len(log_m_k)))
        p_m = Categorical(m_r_stack.view(-1, len(log_m_k)))
        kl_m_ppc = kl_divergence(q_m, p_m).view(batch_size, -1)
        return kl_m_ppc.sum(dim=1)

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        z_batched = Normal(0, 1).sample((batch_size * K_steps, self.comp_vae.ldim))
        x_hat_batched = self.comp_vae.decode(z_batched)
        x_r_batched = x_hat_batched[:, :3, :, :]
        m_r_logids_batched = x_hat_batched[:, 3:, :, :]
        if self.pixel_bound:
            x_r_batched = torch.sigmoid(x_r_batched)
        x_r_k = torch.chunk(x_r_batched, K_steps, dim=0)
        m_r_logits_k = torch.chunk(m_r_logids_batched, K_steps, dim=0)
        m_r_stack = self.get_mask_recon_stack(m_r_logits_k, self.prior_mode, log=False)
        x_r_stack = torch.stack(x_r_k, dim=4)
        gen_image = (m_r_stack * x_r_stack).sum(dim=4)
        log_m_r_k = [item.squeeze(dim=4) for item in torch.split(m_r_stack.log(), 1, dim=4)]
        stats = AttrDict(gen_image=gen_image, x_k=x_r_k, log_m_k=log_m_r_k, mx_k=[(x * m.exp()) for x, m in zip(x_r_k, log_m_r_k)])
        return gen_image, stats


class GenesisV2(nn.Module):

    def __init__(self, cfg):
        super(GenesisV2, self).__init__()
        self.K_steps = cfg.K_steps
        self.pixel_bound = cfg.pixel_bound
        self.feat_dim = cfg.feat_dim
        self.klm_loss = cfg.klm_loss
        self.detach_mr_in_klm = cfg.detach_mr_in_klm
        self.dynamic_K = cfg.dynamic_K
        self.debug = cfg.debug
        self.multi_gpu = cfg.multi_gpu
        self.encoder = UNet(num_blocks=int(np.log2(cfg.img_size) - 1), img_size=cfg.img_size, filter_start=min(cfg.feat_dim, 64), in_chnls=3, out_chnls=cfg.feat_dim, norm='gn')
        self.encoder.final_conv = nn.Identity()
        self.att_process = attention.InstanceColouringSBP(img_size=cfg.img_size, kernel=cfg.kernel, colour_dim=8, K_steps=self.K_steps, feat_dim=cfg.feat_dim, semiconv=cfg.semiconv)
        self.seg_head = B.ConvGNReLU(cfg.feat_dim, cfg.feat_dim, 3, 1, 1)
        self.feat_head = nn.Sequential(B.ConvGNReLU(cfg.feat_dim, cfg.feat_dim, 3, 1, 1), nn.Conv2d(cfg.feat_dim, 2 * cfg.feat_dim, 1))
        self.z_head = nn.Sequential(nn.LayerNorm(2 * cfg.feat_dim), nn.Linear(2 * cfg.feat_dim, 2 * cfg.feat_dim), nn.ReLU(inplace=True), nn.Linear(2 * cfg.feat_dim, 2 * cfg.feat_dim))
        c = cfg.feat_dim
        self.decoder_module = nn.Sequential(B.BroadcastLayer(cfg.img_size // 16), nn.ConvTranspose2d(cfg.feat_dim + 2, c, 5, 2, 2, 1), nn.GroupNorm(8, c), nn.ReLU(inplace=True), nn.ConvTranspose2d(c, c, 5, 2, 2, 1), nn.GroupNorm(8, c), nn.ReLU(inplace=True), nn.ConvTranspose2d(c, min(c, 64), 5, 2, 2, 1), nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True), nn.ConvTranspose2d(min(c, 64), min(c, 64), 5, 2, 2, 1), nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True), nn.Conv2d(min(c, 64), 4, 1))
        self.autoreg_prior = cfg.autoreg_prior
        self.prior_lstm, self.prior_linear = None, None
        if self.autoreg_prior and self.K_steps > 1:
            self.prior_lstm = nn.LSTM(cfg.feat_dim, 4 * cfg.feat_dim)
            self.prior_linear = nn.Linear(4 * cfg.feat_dim, 2 * cfg.feat_dim)
        assert cfg.pixel_std1 == cfg.pixel_std2
        self.std = cfg.pixel_std1

    def forward(self, x):
        batch_size, _, H, W = x.shape
        enc_feat, _ = self.encoder(x)
        enc_feat = F.relu(enc_feat)
        if self.dynamic_K:
            if batch_size > 1:
                log_m_k = [[] for _ in range(self.K_steps)]
                att_stats, log_s_k = None, None
                for f in torch.split(enc_feat, 1, dim=0):
                    log_m_k_b, _, _ = self.att_process(self.seg_head(f), self.K_steps - 1, dynamic_K=True)
                    for step in range(self.K_steps):
                        if step < len(log_m_k_b):
                            log_m_k[step].append(log_m_k_b[step])
                        else:
                            log_m_k[step].append(-10000000000.0 * torch.ones([1, 1, H, W]))
                for step in range(self.K_steps):
                    log_m_k[step] = torch.cat(log_m_k[step], dim=0)
                if self.debug:
                    assert len(log_m_k) == self.K_steps
            else:
                log_m_k, log_s_k, att_stats = self.att_process(self.seg_head(enc_feat), self.K_steps - 1, dynamic_K=True)
        else:
            log_m_k, log_s_k, att_stats = self.att_process(self.seg_head(enc_feat), self.K_steps - 1, dynamic_K=False)
            if self.debug:
                assert len(log_m_k) == self.K_steps
        comp_stats = AttrDict(mu_k=[], sigma_k=[], z_k=[], kl_l_k=[], q_z_k=[])
        for log_m in log_m_k:
            mask = log_m.exp()
            obj_feat = mask * self.feat_head(enc_feat)
            obj_feat = obj_feat.sum((2, 3))
            obj_feat = obj_feat / (mask.sum((2, 3)) + 1e-05)
            mu, sigma_ps = self.z_head(obj_feat).chunk(2, dim=1)
            sigma = B.to_sigma(sigma_ps)
            q_z = Normal(mu, sigma)
            z = q_z.rsample()
            comp_stats['mu_k'].append(mu)
            comp_stats['sigma_k'].append(sigma)
            comp_stats['z_k'].append(z)
            comp_stats['q_z_k'].append(q_z)
        recon, x_r_k, log_m_r_k = self.decode_latents(comp_stats.z_k)
        losses = AttrDict()
        losses['err'] = Genesis.x_loss(x, log_m_r_k, x_r_k, self.std)
        mx_r_k = [(x * logm.exp()) for x, logm in zip(x_r_k, log_m_r_k)]
        if self.klm_loss:
            if self.detach_mr_in_klm:
                log_m_r_k = [m.detach() for m in log_m_r_k]
            losses['kl_m'] = MONet.kl_m_loss(log_m_k=log_m_k, log_m_r_k=log_m_r_k, debug=self.debug)
        losses['kl_l_k'], p_z_k = Genesis.mask_latent_loss(comp_stats.q_z_k, comp_stats.z_k, prior_lstm=self.prior_lstm, prior_linear=self.prior_linear, debug=self.debug)
        stats = AttrDict(recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k, log_m_r_k=log_m_r_k, mx_r_k=mx_r_k, instance_seg=torch.argmax(torch.cat(log_m_k, dim=1), dim=1), instance_seg_r=torch.argmax(torch.cat(log_m_r_k, dim=1), dim=1))
        if self.debug:
            if not self.dynamic_K:
                assert len(log_m_k) == self.K_steps
                assert len(log_m_r_k) == self.K_steps
            misc.check_log_masks(log_m_k)
            misc.check_log_masks(log_m_r_k)
        if self.multi_gpu:
            del comp_stats['q_z_k']
        return recon, losses, stats, att_stats, comp_stats

    def decode_latents(self, z_k):
        x_r_k, m_r_logits_k = [], []
        for z in z_k:
            dec = self.decoder_module(z)
            x_r_k.append(dec[:, :3, :, :])
            m_r_logits_k.append(dec[:, 3:, :, :])
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]
        log_m_r_stack = MONet.get_mask_recon_stack(m_r_logits_k, 'softmax', log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        recon = (m_r_stack * x_r_stack).sum(dim=4)
        return recon, x_r_k, log_m_r_k

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        if self.autoreg_prior:
            z_k = [Normal(0, 1).sample([batch_size, self.feat_dim])]
            state = None
            for k in range(1, K_steps):
                lstm_out, state = self.prior_lstm(z_k[-1].view(1, batch_size, -1), state)
                linear_out = self.prior_linear(lstm_out)
                linear_out = torch.chunk(linear_out, 2, dim=2)
                linear_out = [item.squeeze(0) for item in linear_out]
                mu = torch.tanh(linear_out[0])
                sigma = B.to_prior_sigma(linear_out[1])
                p_z = Normal(mu.view([batch_size, self.feat_dim]), sigma.view([batch_size, self.feat_dim]))
                z_k.append(p_z.sample())
        else:
            p_z = Normal(0, 1)
            z_k = [p_z.sample([batch_size, self.feat_dim]) for _ in range(K_steps)]
        recon, x_r_k, log_m_r_k = self.decode_latents(z_k)
        stats = AttrDict(x_k=x_r_k, log_m_k=log_m_r_k, mx_k=[(x * m.exp()) for x, m in zip(x_r_k, log_m_r_k)])
        return recon, stats


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def to_sigma(x):
    return F.softplus(x + 0.5) + 1e-08


def to_var(x):
    return to_sigma(x) ** 2


class ToVar(nn.Module):

    def __init__(self):
        super(ToVar, self).__init__()

    def forward(self, x):
        return to_var(x)


class GatedConvTranspose2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding=0, dilation=1, activation=None, h_norm=None, g_norm=None):
        super(GatedConvTranspose2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.ConvTranspose2d(input_channels, 2 * output_channels, kernel_size, stride, padding, output_padding, dilation=dilation)
        self.h_norm, self.g_norm = None, None
        if h_norm == 'in':
            self.h_norm = nn.InstanceNorm2d(output_channels, affine=True)
        elif h_norm == 'bn':
            self.h_norm = nn.BatchNorm2d(output_channels)
        elif h_norm is None or h_norm == 'none':
            pass
        else:
            raise ValueError('Normalisation option not recognised.')
        if g_norm == 'in':
            self.g_norm = nn.InstanceNorm2d(output_channels, affine=True)
        elif g_norm == 'bn':
            self.g_norm = nn.BatchNorm2d(output_channels)
        elif g_norm is None or g_norm == 'none':
            pass
        else:
            raise ValueError('Normalisation option not recognised.')

    def forward(self, x):
        h, g = torch.chunk(self.conv(x), 2, dim=1)
        if self.h_norm is not None:
            h = self.h_norm(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.g_norm is not None:
            g = self.g_norm(g)
        g = self.sigmoid(g)
        return h * g


def build_gc_decoder(cin, cout, stride, zdim, kz, hn=None, gn=None):
    assert len(cin) == len(cout) and len(cin) == len(stride)
    layers = [GatedConvTranspose2d(zdim, cin[0], kz, 1, 0)]
    for l, (i, o, s) in enumerate(zip(cin, cout, stride)):
        layers.append(GatedConvTranspose2d(i, o, 5, s, 2, s - 1, h_norm=hn, g_norm=gn))
    return nn.Sequential(*layers)


class GatedConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None, h_norm=None, g_norm=None):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(input_channels, 2 * output_channels, kernel_size, stride, padding, dilation)
        self.h_norm, self.g_norm = None, None
        if h_norm == 'in':
            self.h_norm = nn.InstanceNorm2d(output_channels, affine=True)
        elif h_norm == 'bn':
            self.h_norm = nn.BatchNorm2d(output_channels)
        elif h_norm is None or h_norm == 'none':
            pass
        else:
            raise ValueError('Normalisation option not recognised.')
        if g_norm == 'in':
            self.g_norm = nn.InstanceNorm2d(output_channels, affine=True)
        elif g_norm == 'bn':
            self.g_norm = nn.BatchNorm2d(output_channels)
        elif g_norm is None or g_norm == 'none':
            pass
        else:
            raise ValueError('Normalisation option not recognised.')

    def forward(self, x):
        h, g = torch.chunk(self.conv(x), 2, dim=1)
        if self.h_norm is not None:
            h = self.h_norm(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.g_norm is not None:
            g = self.g_norm(g)
        g = self.sigmoid(g)
        return h * g


def build_gc_encoder(cin, cout, stride, cfc, kfc, hn=None, gn=None):
    assert len(cin) == len(cout) and len(cin) == len(stride)
    layers = []
    for l, (i, o, s) in enumerate(zip(cin, cout, stride)):
        layers.append(GatedConv2d(i, o, 5, s, 2, h_norm=hn, g_norm=gn))
    layers.append(GatedConv2d(cout[-1], cfc, kfc, 1, 0))
    return nn.Sequential(*layers)


class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, z_size, input_size, nout, enc_norm=None, dec_norm=None):
        super(VAE, self).__init__()
        self.z_size = z_size
        self.input_size = input_size
        if nout is not None:
            self.nout = nout
        else:
            self.nout = input_size[0]
        self.enc_norm = enc_norm
        self.dec_norm = dec_norm
        if self.input_size[1] == 32 and self.input_size[2] == 32:
            self.last_kernel_size = 8
            strides = [1, 2, 1, 2, 1]
        elif self.input_size[1] == 64 and self.input_size[2] == 64:
            self.last_kernel_size = 16
            strides = [1, 2, 1, 2, 1]
        elif self.input_size[1] == 128 and self.input_size[2] == 128:
            self.last_kernel_size = 16
            strides = [2, 2, 2, 1, 1]
        elif self.input_size[1] == 256 and self.input_size[2] == 256:
            self.last_kernel_size = 16
            strides = [2, 2, 2, 2, 1]
        else:
            raise ValueError('Invalid input size.')
        self.q_z_nn_output_dim = 256
        cin = [self.input_size[0], 32, 32, 64, 64]
        cout = [32, 32, 64, 64, 64]
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder(cin, cout, strides)
        cin = [64, 64, 32, 32, 32]
        cout = [64, 32, 32, 32, 32]
        self.p_x_nn, self.p_x_mean = self.create_decoder(cin, cout, list(reversed(strides)))
        self.log_det_j = torch.tensor(0)

    def create_encoder(self, cin, cout, strides):
        """
        Helper function to create the elemental blocks for the encoder.
        Creates a gated convnet encoder.
        the encoder expects data as input of shape:
        (batch_size, num_channels, width, height).
        """
        q_z_nn = build_gc_encoder(cin, cout, strides, self.q_z_nn_output_dim, self.last_kernel_size, hn=self.enc_norm, gn=self.enc_norm)
        q_z_mean = nn.Linear(256, self.z_size)
        q_z_var = nn.Sequential(nn.Linear(256, self.z_size), ToVar())
        return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self, cin, cout, strides):
        """
        Helper function to create the elemental blocks for the decoder.
        Creates a gated convnet decoder.
        """
        p_x_nn = build_gc_decoder(cin, cout, strides, self.z_size, self.last_kernel_size, hn=self.dec_norm, gn=self.dec_norm)
        p_x_mean = nn.Conv2d(cout[-1], self.nout, 1, 1, 0)
        return p_x_nn, p_x_mean

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        q_z = Normal(mu, var.sqrt())
        z = q_z.rsample()
        return z, q_z

    def encode(self, x):
        """
        Encoder expects following data shapes as input:
        shape = (batch_size, num_channels, width, height)
        """
        h = self.q_z_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """
        z = z.view(z.size(0), self.z_size, 1, 1)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """
        z_mu, z_var = self.encode(x)
        z, q_z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)
        stats = AttrDict(x=x_mean, mu=z_mu, sigma=z_var.sqrt(), z=z)
        return x_mean, stats


class BaselineVAE(nn.Module):

    def __init__(self, cfg):
        super(BaselineVAE, self).__init__()
        cfg.K_steps = None
        self.ldim = cfg.latent_dimension
        self.pixel_std = cfg.pixel_std
        self.pixel_bound = cfg.pixel_bound
        self.debug = cfg.debug
        nin = cfg.input_channels if hasattr(cfg, 'input_channels') else 3
        self.vae = VAE(self.ldim, [nin, cfg.img_size, cfg.img_size], nin)
        if cfg.broadcast_decoder:
            self.vae.p_x_nn = nn.Sequential(Flatten(), BroadcastDecoder(in_chnls=self.ldim, out_chnls=64, h_chnls=64, num_layers=4, img_dim=cfg.img_size, act=nn.ELU()), nn.ELU())
            self.vae.p_x_mean = nn.Conv2d(64, nin, 1, 1, 0)

    def forward(self, x):
        """ x (torch.Tensor): Input images [batch size, 3, dim, dim] """
        recon, stats = self.vae(x)
        if self.pixel_bound:
            recon = torch.sigmoid(recon)
        p_xr = Normal(recon, self.pixel_std)
        err = -p_xr.log_prob(x).sum(dim=(1, 2, 3))
        p_z = Normal(0, 1)
        if 'z' in stats:
            q_z = Normal(stats.mu, stats.sigma)
            kl = q_z.log_prob(stats.z) - p_z.log_prob(stats.z)
            kl = kl.sum(dim=1)
        else:
            q_z_0 = Normal(stats.mu_0, stats.sigma_0)
            kl = q_z_0.log_prob(stats.z_0) - p_z.log_prob(stats.z_k)
            kl = kl.sum(dim=1) - stats.ldj
        losses = AttrDict(err=err, kl_l=kl)
        return recon, losses, stats, None, None

    def sample(self, batch_size, *args, **kwargs):
        z = Normal(0, 1).sample([batch_size, self.ldim])
        x = self.vae.decode(z)
        if self.pixel_bound:
            x = torch.sigmoid(x)
        return x, AttrDict(z=z)

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, stats, _, _ = self.forward(image_batch)
        return stats.z


class SimpleSBP(nn.Module):

    def __init__(self, core):
        super(SimpleSBP, self).__init__()
        self.core = core

    def forward(self, x, steps_to_run):
        log_m_k = []
        log_s_k = [torch.zeros_like(x)[:, :1, :, :]]
        for step in range(steps_to_run):
            core_out, _ = self.core(torch.cat((x, log_s_k[step]), dim=1))
            a_logits = core_out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            log_m_k.append(log_s_k[step] + log_a)
            log_s_k.append(log_s_k[step] + log_neg_a)
        log_m_k.append(log_s_k[-1])
        return log_m_k, log_s_k, {}

    def masks_from_zm_k(self, zm_k, img_size):
        b_sz = zm_k[0].size(0)
        log_m_k = []
        log_s_k = [torch.zeros(b_sz, 1, img_size, img_size)]
        other_k = []
        for zm in zm_k:
            core_out = self.core.decode(zm)
            a_logits = core_out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            other_k.append(core_out[:, 1:, :, :])
            log_m_k.append(log_s_k[-1] + log_a)
            log_s_k.append(log_s_k[-1] + log_neg_a)
        log_m_k.append(log_s_k[-1])
        return log_m_k, log_s_k, other_k


class LatentSBP(SimpleSBP):

    def __init__(self, core):
        super(LatentSBP, self).__init__(core)
        self.lstm = nn.LSTM(core.z_size + 256, 2 * core.z_size)
        self.linear = nn.Linear(2 * core.z_size, 2 * core.z_size)

    def forward(self, x, steps_to_run):
        h = self.core.q_z_nn(x)
        bs = h.size(0)
        h = h.view(bs, -1)
        mean_0 = self.core.q_z_mean(h)
        var_0 = self.core.q_z_var(h)
        z, q_z = self.core.reparameterize(mean_0, var_0)
        z_k = [z]
        q_z_k = [q_z]
        state = None
        for step in range(1, steps_to_run):
            h_and_z = torch.cat([h, z_k[-1]], dim=1)
            lstm_out, state = self.lstm(h_and_z.view(1, bs, -1), state)
            linear_out = self.linear(lstm_out)[0, :, :]
            linear_out = torch.chunk(linear_out, 2, dim=1)
            mean_k = linear_out[0]
            var_k = B.to_var(linear_out[1])
            z, q_z = self.core.reparameterize(mean_k, var_k)
            z_k.append(z)
            q_z_k.append(q_z)
        log_m_k = []
        stats_k = []
        log_s_k = [torch.zeros_like(x)[:, :1, :, :]]
        z_batch = torch.cat(z_k, dim=0)
        core_out_batch = self.core.decode(z_batch)
        core_out = torch.chunk(core_out_batch, steps_to_run, dim=0)
        for step, (z, q_z, out) in enumerate(zip(z_k, q_z_k, core_out)):
            stats = AttrDict(x=out, mu=q_z.mean, sigma=q_z.scale, z=z)
            a_logits = out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            log_m_k.append(log_s_k[step] + log_a)
            log_s_k.append(log_s_k[step] + log_neg_a)
            stats_k.append(stats)
        log_m_k.append(log_s_k[-1])
        stats = AttrDict()
        for key in stats_k[0]:
            stats[key + '_k'] = [s[key] for s in stats_k]
        return log_m_k, log_s_k, stats


class InstanceColouringSBP(nn.Module):

    def __init__(self, img_size, kernel='gaussian', colour_dim=8, K_steps=None, feat_dim=None, semiconv=True):
        super(InstanceColouringSBP, self).__init__()
        self.img_size = img_size
        self.kernel = kernel
        self.colour_dim = colour_dim
        if self.kernel == 'laplacian':
            sigma_init = 1.0 / (np.sqrt(K_steps) * np.log(2))
        elif self.kernel == 'gaussian':
            sigma_init = 1.0 / (K_steps * np.log(2))
        elif self.kernel == 'epanechnikov':
            sigma_init = 2.0 / K_steps
        else:
            return ValueError('No valid kernel.')
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())
        if semiconv:
            self.colour_head = B.SemiConv(feat_dim, self.colour_dim, img_size)
        else:
            self.colour_head = nn.Conv2d(feat_dim, self.colour_dim, 1)

    def forward(self, features, steps_to_run, debug=False, dynamic_K=False, *args, **kwargs):
        batch_size = features.size(0)
        stats = AttrDict()
        if isinstance(features, tuple):
            features = features[0]
        if dynamic_K:
            assert batch_size == 1
        colour_out = self.colour_head(features)
        if isinstance(colour_out, tuple):
            colour, delta = colour_out
        else:
            colour, delta = colour_out, None
        rand_pixel = torch.empty(batch_size, 1, *colour.shape[2:])
        rand_pixel = rand_pixel.uniform_()
        seed_list = []
        log_m_k = []
        log_s_k = [torch.zeros(batch_size, 1, self.img_size, self.img_size)]
        for step in range(steps_to_run):
            scope = F.interpolate(log_s_k[step].exp(), size=colour.shape[2:], mode='bilinear', align_corners=False)
            pixel_probs = rand_pixel * scope
            rand_max = pixel_probs.flatten(2).argmax(2).flatten()
            seed = torch.empty((batch_size, self.colour_dim))
            for bidx in range(batch_size):
                seed[bidx, :] = colour.flatten(2)[bidx, :, rand_max[bidx]]
            seed_list.append(seed)
            if self.kernel == 'laplacian':
                distance = B.euclidian_distance(colour, seed)
                alpha = torch.exp(-distance / self.log_sigma.exp())
            elif self.kernel == 'gaussian':
                distance = B.squared_distance(colour, seed)
                alpha = torch.exp(-distance / self.log_sigma.exp())
            elif self.kernel == 'epanechnikov':
                distance = B.squared_distance(colour, seed)
                alpha = (1 - distance / self.log_sigma.exp()).relu()
            else:
                raise ValueError('No valid kernel.')
            alpha = alpha.unsqueeze(1)
            if debug:
                assert alpha.max() <= 1, alpha.max()
                assert alpha.min() >= 0, alpha.min()
            alpha = B.clamp_preserve_gradients(alpha, 0.01, 0.99)
            log_a = torch.log(alpha)
            log_neg_a = torch.log(1 - alpha)
            log_m = log_s_k[step] + log_a
            if dynamic_K and log_m.exp().sum() < 20:
                break
            log_m_k.append(log_m)
            log_s_k.append(log_s_k[step] + log_neg_a)
        log_m_k.append(log_s_k[-1])
        stats.update({'colour': colour, 'delta': delta, 'seeds': seed_list})
        return log_m_k, log_s_k, stats


class ToSigma(nn.Module):

    def __init__(self):
        super(ToSigma, self).__init__()

    def forward(self, x):
        return to_sigma(x)


class ScalarGate(nn.Module):

    def __init__(self, init=0.0):
        super(ScalarGate, self).__init__()
        self.gate = nn.Parameter(torch.tensor(init))

    def forward(self, x):
        return self.gate * x


class UnFlatten(nn.Module):

    def __init__(self):
        super(UnFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)


class PixelCoords(nn.Module):

    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim), torch.linspace(-1, 1, im_dim))
        self.g_1 = g_1.view((1, 1) + g_1.shape)
        self.g_2 = g_2.view((1, 1) + g_2.shape)

    def forward(self, x):
        g_1 = self.g_1.expand(x.size(0), -1, -1, -1)
        g_2 = self.g_2.expand(x.size(0), -1, -1, -1)
        return torch.cat((x, g_1, g_2), dim=1)


class BroadcastLayer(nn.Module):

    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        self.pixel_coords = PixelCoords(dim)

    def forward(self, x):
        b_sz = x.size(0)
        if x.dim() == 2:
            x = x.view(b_sz, -1, 1, 1)
            x = x.expand(-1, -1, self.dim, self.dim)
        else:
            x = F.interpolate(x, self.dim)
        return self.pixel_coords(x)


class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class ConvReLU(nn.Sequential):

    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__(nn.Conv2d(nin, nout, kernel, stride, padding), nn.ReLU(inplace=True))


class ConvINReLU(nn.Sequential):

    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvINReLU, self).__init__(nn.Conv2d(nin, nout, kernel, stride, padding, bias=False), nn.InstanceNorm2d(nout, affine=True), nn.ReLU(inplace=True))


class ConvGNReLU(nn.Sequential):

    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(nn.Conv2d(nin, nout, kernel, stride, padding, bias=False), nn.GroupNorm(groups, nout), nn.ReLU(inplace=True))


def pixel_coords(img_size):
    g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, img_size), torch.linspace(-1, 1, img_size))
    g_1 = g_1.view(1, 1, img_size, img_size)
    g_2 = g_2.view(1, 1, img_size, img_size)
    return torch.cat((g_1, g_2), dim=1)


class SemiConv(nn.Module):

    def __init__(self, nin, nout, img_size):
        super(SemiConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 1)
        self.gate = ScalarGate()
        coords = pixel_coords(img_size)
        zeros = torch.zeros(1, nout - 2, img_size, img_size)
        self.uv = torch.cat((zeros, coords), dim=1)

    def forward(self, x):
        out = self.gate(self.conv(x))
        delta = out[:, -2:, :, :]
        return out + self.uv, delta


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BroadcastLayer,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvReLU,
     lambda: ([], {'nin': 4, 'nout': 4, 'kernel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedConv2d,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelCoords,
     lambda: ([], {'im_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalarGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SemiConv,
     lambda: ([], {'nin': 4, 'nout': 4, 'img_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleSBP,
     lambda: ([], {'core': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (ToSigma,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ToVar,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnFlatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_applied_ai_lab_genesis(_paritybench_base):
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

