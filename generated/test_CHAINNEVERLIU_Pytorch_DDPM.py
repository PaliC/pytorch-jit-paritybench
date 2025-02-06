import sys
_module = sys.modules[__name__]
del sys
simpleDiffusion = _module
varianceSchedule = _module
image_test = _module
UNet = _module
networkHelper = _module
trainNetworkHelper = _module

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


import torchvision


import torch


import torchvision.transforms as transforms


from torch.optim import Adam


from functools import partial


from torch import nn


from torch import einsum


import torch.nn.functional as F


import math


from inspect import isfunction


from torchvision.transforms import Compose


from torchvision.transforms import Lambda


from torchvision.transforms import ToPILImage


import torch.nn as nn


def cosine_beta_schedule(timesteps, s=0.008, **kwargs):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class VarianceSchedule(nn.Module):

    def __init__(self, schedule_name='linear_beta_schedule', beta_start=None, beta_end=None):
        super(VarianceSchedule, self).__init__()
        self.schedule_name = schedule_name
        beta_schedule_dict = {'linear_beta_schedule': linear_beta_schedule, 'cosine_beta_schedule': cosine_beta_schedule, 'quadratic_beta_schedule': quadratic_beta_schedule, 'sigmoid_beta_schedule': sigmoid_beta_schedule}
        if schedule_name in beta_schedule_dict:
            self.selected_schedule = beta_schedule_dict[schedule_name]
        else:
            raise ValueError('Function not found in dictionary')
        if beta_end and beta_start is None and schedule_name != 'cosine_beta_schedule':
            self.beta_start = 0.0001
            self.beta_end = 0.02
        else:
            self.beta_start = beta_start
            self.beta_end = beta_end

    def forward(self, timesteps):
        return self.selected_schedule(timesteps=timesteps) if self.schedule_name == 'cosine_beta_schedule' else self.selected_schedule(timesteps=timesteps, beta_start=self.beta_start, beta_end=self.beta_end)


def extract(a, t, x_shape):
    """
    从给定的张量a中检索特定的元素。t是一个包含要检索的索引的张量，
    这些索引对应于a张量中的元素。这个函数的输出是一个张量，
    包含了t张量中每个索引对应的a张量中的元素
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionModel(nn.Module):

    def __init__(self, schedule_name='linear_beta_schedule', timesteps=1000, beta_start=0.0001, beta_end=0.02, denoise_model=None):
        super(DiffusionModel, self).__init__()
        self.denoise_model = denoise_model
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type='l1'):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))

    def forward(self, mode, **kwargs):
        if mode == 'train':
            if 'x_start' and 't' in kwargs.keys():
                if 'loss_type' and 'noise' in kwargs.keys():
                    return self.compute_loss(x_start=kwargs['x_start'], t=kwargs['t'], noise=kwargs['noise'], loss_type=kwargs['loss_type'])
                elif 'loss_type' in kwargs.keys():
                    return self.compute_loss(x_start=kwargs['x_start'], t=kwargs['t'], loss_type=kwargs['loss_type'])
                elif 'noise' in kwargs.keys():
                    return self.compute_loss(x_start=kwargs['x_start'], t=kwargs['t'], noise=kwargs['noise'])
                else:
                    return self.compute_loss(x_start=kwargs['x_start'], t=kwargs['t'])
            else:
                raise ValueError('扩散模型在训练时必须传入参数x_start和t！')
        elif mode == 'generate':
            if 'image_size' and 'batch_size' and 'channels' in kwargs.keys():
                return self.sample(image_size=kwargs['image_size'], batch_size=kwargs['batch_size'], channels=kwargs['channels'])
            else:
                raise ValueError('扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数')
        else:
            raise ValueError('mode参数必须从{train}和{generate}两种模式中选择')


class WeightStandardizedConv2d(nn.Conv2d):
    """
    权重标准化后的卷积模块
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-05 if x.dtype == torch.float32 else 0.001
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def exists(x):
    """
    判断数值是否为空
    :param x: 输入数据
    :return: 如果不为空则True 反之则返回False
    """
    return x is not None


class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def default(val, d):
    """
    该函数的目的是提供一个简单的机制来获取给定变量的默认值。
    如果 val 存在，则返回该值。如果不存在，则使用 d 函数提供的默认值，
    或者如果 d 不是一个函数，则返回 d。
    :param val:需要判断的变量
    :param d:提供默认值的变量或函数
    :return:
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Downsample(dim, dim_out=None):
    """
    下采样模块的作用是将输入张量的分辨率降低，通常用于在深度学习模型中对特征图进行降采样。
    在这个实现中，下采样操作的方式是使用一个 $2 	imes 2$ 的最大池化操作，
    将输入张量的宽和高都缩小一半，然后再使用上述的变换和卷积操作得到输出张量。
    由于这个实现使用了形状变换操作，因此没有使用传统的卷积或池化操作进行下采样，
    从而避免了在下采样过程中丢失信息的问题。
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2), nn.Conv2d(dim * 4, default(dim_out, dim), 1))


class Residual(nn.Module):

    def __init__(self, fn):
        """
        残差连接模块
        :param fn: 激活函数类型
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        残差连接前馈
        :param x: 输入数据
        :param args:
        :param kwargs:
        :return: f(x) + x
        """
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def Upsample(dim, dim_out=None):
    """
    这个上采样模块的作用是将输入张量的尺寸在宽和高上放大 2 倍
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))


class Unet(nn.Module):

    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, self_condition=False, resnet_block_groups=4):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(nn.ModuleList([block_klass(dim_in, dim_in, time_emb_dim=time_dim), block_klass(dim_in, dim_in, time_emb_dim=time_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(nn.ModuleList([block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim), block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim), Residual(PreNorm(dim_out, LinearAttention(dim_out))), Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)]))
        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            None
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            None
        torch.save(model.state_dict(), path + '/' + 'checkpoints.pth')
        self.val_loss_min = val_loss


class TrainerBase(nn.Module):

    def __init__(self, epoches, train_loader, optimizer, device, IFEarlyStopping, IFadjust_learning_rate, **kwargs):
        super(TrainerBase, self).__init__()
        self.epoches = epoches
        if self.epoches is None:
            raise ValueError('请传入训练总迭代次数')
        self.train_loader = train_loader
        if self.train_loader is None:
            raise ValueError('请传入train_loader')
        self.optimizer = optimizer
        if self.optimizer is None:
            raise ValueError('请传入优化器类')
        self.device = device
        if self.device is None:
            raise ValueError('请传入运行设备类型')
        self.IFEarlyStopping = IFEarlyStopping
        if IFEarlyStopping:
            if 'patience' in kwargs.keys():
                self.early_stopping = EarlyStopping(patience=kwargs['patience'], verbose=True)
            else:
                raise ValueError('启用提前停止策略必须输入{patience=int X}参数')
            if 'val_loader' in kwargs.keys():
                self.val_loader = kwargs['val_loader']
            else:
                raise ValueError('启用提前停止策略必须输入验证集val_loader')
        self.IFadjust_learning_rate = IFadjust_learning_rate
        if IFadjust_learning_rate:
            if 'types' in kwargs.keys():
                self.types = kwargs['types']
                if 'lr_adjust' in kwargs.keys():
                    self.lr_adjust = kwargs['lr_adjust']
                else:
                    self.lr_adjust = None
            else:
                raise ValueError('启用学习率调整策略必须从{type1 or type2}中选择学习率调整策略参数types')

    def adjust_learning_rate(self, epoch, learning_rate):
        if self.types == 'type1':
            lr_adjust = {epoch: learning_rate * 0.1 ** ((epoch - 1) // 10)}
        elif self.types == 'type2':
            if self.lr_adjust is not None:
                lr_adjust = self.lr_adjust
            else:
                lr_adjust = {(5): 0.0001, (10): 5e-05, (20): 1e-05, (25): 5e-06, (30): 1e-06, (35): 5e-07, (40): 1e-08}
        else:
            raise ValueError('请从{{0}or{1}}中选择学习率调整策略参数types'.format('type1', 'type2'))
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            None

    @staticmethod
    def save_best_model(model, path):
        torch.save(model.state_dict(), path + '/' + 'BestModel.pth')
        None

    def forward(self, model, *args, **kwargs):
        pass


class SimpleDiffusionTrainer(TrainerBase):

    def __init__(self, epoches=None, train_loader=None, optimizer=None, device=None, IFEarlyStopping=False, IFadjust_learning_rate=False, **kwargs):
        super(SimpleDiffusionTrainer, self).__init__(epoches, train_loader, optimizer, device, IFEarlyStopping, IFadjust_learning_rate, **kwargs)
        if 'timesteps' in kwargs.keys():
            self.timesteps = kwargs['timesteps']
        else:
            raise ValueError('扩散模型训练必须提供扩散步数参数')

    def forward(self, model, *args, **kwargs):
        for i in range(self.epoches):
            losses = []
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for step, (features, labels) in loop:
                features = features
                batch_size = features.shape[0]
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                loss = model(mode='train', x_start=features, t=t, loss_type='huber')
                losses.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loop.set_description(f'Epoch [{i}/{self.epoches}]')
                loop.set_postfix(loss=loss.item())
        if 'model_save_path' in kwargs.keys():
            self.save_best_model(model=model, path=kwargs['model_save_path'])
        return model


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_CHAINNEVERLIU_Pytorch_DDPM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

