import sys
_module = sys.modules[__name__]
del sys
conf = _module
iunets = _module
baseline_networks = _module
cayley = _module
expm = _module
householder = _module
layers = _module
networks = _module
utils = _module
setup = _module

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


from torch import nn


from torch.autograd import Function


import numpy as np


from warnings import warn


from typing import Callable


from typing import Union


from typing import Iterable


from typing import Tuple


from torch import Tensor


from torch.nn.common_types import _size_1_t


from torch.nn.common_types import _size_2_t


from torch.nn.common_types import _size_3_t


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


import torch.nn.functional as F


import warnings


from typing import Any


from typing import Sized


from typing import List


from typing import Optional


class StandardBlock(nn.Module):

    def __init__(self, dim, num_in_channels, num_out_channels, depth=2, zero_init=False, normalization='instance', **kwargs):
        super(StandardBlock, self).__init__()
        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]
        self.seq = nn.ModuleList()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        for i in range(depth):
            current_in_channels = max(num_in_channels, num_out_channels)
            current_out_channels = max(num_in_channels, num_out_channels)
            if i == 0:
                current_in_channels = num_in_channels
            if i == depth - 1:
                current_out_channels = num_out_channels
            self.seq.append(conv_op(current_in_channels, current_out_channels, 3, padding=1, bias=False))
            torch.nn.init.kaiming_uniform_(self.seq[-1].weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
            if normalization == 'instance':
                norm_op = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][dim - 1]
                self.seq.append(norm_op(current_out_channels, affine=True))
            elif normalization == 'group':
                self.seq.append(nn.GroupNorm(np.min(1, current_out_channels // 8), current_out_channels, affine=True))
            elif normalization == 'batch':
                norm_op = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dim - 1]
                self.seq.append(norm_op(current_out_channels, eps=0.001))
            else:
                None
            self.seq.append(nn.LeakyReLU(inplace=True))
        if zero_init:
            torch.nn.init.zeros_(self.seq[-1].weight)
            torch.nn.init.zeros_(self.seq[-1].bias)
        self.F = nn.Sequential(*self.seq)

    def forward(self, x):
        x = self.F(x)
        return x


def get_num_channels(input_shape_or_channels):
    """
    Small helper function which outputs the number of
    channels regardless of whether the input shape or
    the number of channels were passed.
    """
    if hasattr(input_shape_or_channels, '__iter__'):
        return input_shape_or_channels[0]
    else:
        return input_shape_or_channels


class StandardUNet(nn.Module):

    def __init__(self, input_shape_or_channels, dim=None, architecture=[2, 2, 2, 2], base_filters=32, skip_connection=False, block_type=StandardBlock, zero_init=False, *args, **kwargs):
        super(StandardUNet, self).__init__()
        self.input_channels = get_num_channels(input_shape_or_channels)
        self.base_filters = base_filters
        self.architecture = architecture
        self.n_levels = len(self.architecture)
        self.dim = dim
        self.skip_connection = skip_connection
        self.block_type = block_type
        pool_ops = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]
        pool_op = pool_ops[dim - 1]
        upsampling_ops = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
        upsampling_op = upsampling_ops[dim - 1]
        filters = self.base_filters
        filters_list = [filters]
        self.module_L = nn.ModuleList()
        self.module_R = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        for i in range(self.n_levels):
            self.module_L.append(nn.ModuleList())
            self.downsampling_layers.append(pool_op(kernel_size=2))
            depth = architecture[i]
            for j in range(depth):
                if i == 0 and j == 0:
                    in_channels = self.input_channels
                else:
                    in_channels = self.base_filters * 2 ** i
                if j == depth - 1:
                    out_channels = self.base_filters * 2 ** (i + 1)
                else:
                    out_channels = self.base_filters * 2 ** i
                self.module_L[i].append(self.block_type(self.dim, in_channels, out_channels, zero_init, *args, **kwargs))
        for i in range(self.n_levels - 1):
            self.module_R.append(nn.ModuleList())
            depth = architecture[i]
            for j in range(depth):
                if j == 0:
                    in_channels = 3 * self.base_filters * 2 ** (i + 1)
                else:
                    in_channels = self.base_filters * 2 ** (i + 1)
                out_channels = self.base_filters * 2 ** (i + 1)
                self.module_R[i].append(self.block_type(self.dim, in_channels, out_channels, zero_init, *args, **kwargs))
            self.upsampling_layers.append(upsampling_op(self.base_filters * 2 ** (i + 2), self.base_filters * 2 ** (i + 2), kernel_size=2, stride=2))
        if self.skip_connection:
            conv_ops = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
            conv_op = conv_ops[self.dim - 1]
            self.output_layer = conv_op(self.base_filters * 2, self.input_channels, 3, padding=1)

    def forward(self, input, *args, **kwargs):
        skip_inputs = []
        x = input
        for i in range(self.n_levels):
            depth = self.architecture[i]
            for j in range(depth):
                x = self.module_L[i][j](x)
            if i < self.n_levels - 1:
                skip_inputs.append(x)
                x = self.downsampling_layers[i](x)
        for i in range(self.n_levels - 2, -1, -1):
            depth = self.architecture[i]
            x = self.upsampling_layers[i](x)
            y = skip_inputs.pop()
            x = torch.cat((x, y), dim=1)
            for j in range(depth):
                x = self.module_R[i][j](x)
        if self.skip_connection:
            x = self.output_layer(x) + input
        return x


def _cayley(A):
    I = torch.eye(A.shape[-1], device=A.device)
    LU = torch.lu(I + A, pivot=True)
    return torch.lu_solve(I - A, *LU)


def _cayley_frechet(A, H, Q=None):
    I = torch.eye(A.shape[-1], device=A.device)
    if Q is None:
        Q = _cayley(A)
    _LU = torch.lu(I + A, pivot=True)
    p = torch.lu_solve(Q, *_LU)
    _LU = torch.lu(I - A, pivot=True)
    q = torch.lu_solve(H, *_LU)
    return 2.0 * q @ p


class cayley(Function):
    """Computes the Cayley transform.

    """

    @staticmethod
    def forward(ctx, M):
        cayley_M = _cayley(M)
        ctx.save_for_backward(M, cayley_M)
        return cayley_M

    @staticmethod
    def backward(ctx, grad_out):
        M, cayley_M = ctx.saved_tensors
        dcayley_M = _cayley_frechet(M, grad_out, Q=cayley_M)
        return dcayley_M


def __calculate_kernel_matrix_cayley__(weight, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return cayley.apply(skew_symmetric_matrix)


def matrix_1_norm(A):
    """Calculates the 1-norm of a matrix or a batch of matrices.

    Args:
        A (torch.Tensor): Can be either of size (n,n) or (m,n,n).

    Returns:
        torch.Tensor : The 1-norm of A.
    """
    norm, indices = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1)
    return norm


def _compute_scales(A):
    """Compute optimal parameters for scaling-and-squaring algorithm.

    The constants used in this function are determined by the MATLAB
    function found in
    https://github.com/cetmann/pytorch_expm/blob/master/determine_frechet_scaling_constant.m
    """
    norm = matrix_1_norm(A)
    max_norm = torch.max(norm)
    s = torch.zeros_like(norm)
    if A.dtype == torch.float64:
        if A.requires_grad:
            ell = {(3): 0.010813385777848, (5): 0.199806320697895, (7): 0.783460847296204, (9): 1.782448623969279, (13): 4.740307543765127}
        else:
            ell = {(3): 0.014955852179582, (5): 0.253939833006323, (7): 0.950417899616293, (9): 2.097847961257068, (13): 5.371920351148152}
        if max_norm >= ell[9]:
            m = 13
            magic_number = ell[m]
            s = torch.relu_(torch.ceil(torch.log2_(norm / magic_number)))
        else:
            for m in [3, 5, 7, 9]:
                if max_norm < ell[m]:
                    magic_number = ell[m]
                    break
    elif A.dtype == torch.float32:
        if A.requires_grad:
            ell = {(3): 0.30803304184533, (5): 1.482532614793145, (7): 3.248671755200478}
        else:
            ell = {(3): 0.425873001692283, (5): 1.880152677804762, (7): 3.92572478313866}
        if max_norm >= ell[5]:
            m = 7
            magic_number = ell[m]
            s = torch.relu_(torch.ceil(torch.log2_(norm / magic_number)))
        else:
            for m in [3, 5]:
                if max_norm < ell[m]:
                    magic_number = ell[m]
                    break
    return s, m


def _eye_like(M, device=None, dtype=None):
    """Creates an identity matrix of the same shape as another matrix.

    For matrix M, the output is same shape as M, if M is a (n,n)-matrix.
    If M is a batch of m matrices (i.e. a (m,n,n)-tensor), create a batch of
    (n,n)-identity-matrices.

    Args:
        M (torch.Tensor) : A tensor of either shape (n,n) or (m,n,n), for
            which either an identity matrix or a batch of identity matrices
            of the same shape will be created.
        device (torch.device, optional) : The device on which the output
            will be placed. By default, it is placed on the same device
            as M.
        dtype (torch.dtype, optional) : The dtype of the output. By default,
            it is the same dtype as M.

    Returns:
        torch.Tensor : Identity matrix or batch of identity matrices.
    """
    assert len(M.shape) in [2, 3]
    assert M.shape[-1] == M.shape[-2]
    n = M.shape[-1]
    if device is None:
        device = M.device
    if dtype is None:
        dtype = M.dtype
    eye = torch.eye(M.shape[-1], device=device, dtype=dtype)
    if len(M.shape) == 2:
        return eye
    else:
        m = M.shape[0]
        return eye.view(-1, n, n).expand(m, -1, -1)


def _expm_frechet_pade(A, E, m=7):
    assert m in [3, 5, 7, 9, 13]
    if m == 3:
        b = [120.0, 60.0, 12.0, 1.0]
    elif m == 5:
        b = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0]
    elif m == 7:
        b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0]
    elif m == 9:
        b = [17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0]
    elif m == 13:
        b = [6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0]
    I = _eye_like(A)
    if m != 13:
        if m >= 3:
            M_2 = A @ E + E @ A
            A_2 = A @ A
            U = b[3] * A_2
            V = b[2] * A_2
            L_U = b[3] * M_2
            L_V = b[2] * M_2
        if m >= 5:
            M_4 = A_2 @ M_2 + M_2 @ A_2
            A_4 = A_2 @ A_2
            U = U + b[5] * A_4
            V = V + b[4] * A_4
            L_U = L_U + b[5] * M_4
            L_V = L_V + b[4] * M_4
        if m >= 7:
            M_6 = A_4 @ M_2 + M_4 @ A_2
            A_6 = A_4 @ A_2
            U = U + b[7] * A_6
            V = V + b[6] * A_6
            L_U = L_U + b[7] * M_6
            L_V = L_V + b[6] * M_6
        if m == 9:
            M_8 = A_4 @ M_4 + M_4 @ A_4
            A_8 = A_4 @ A_4
            U = U + b[9] * A_8
            V = V + b[8] * A_8
            L_U = L_U + b[9] * M_8
            L_V = L_V + b[8] * M_8
        U = U + b[1] * I
        V = U + b[0] * I
        del I
        L_U = A @ L_U
        L_U = L_U + E @ U
        U = A @ U
    else:
        M_2 = A @ E + E @ A
        A_2 = A @ A
        M_4 = A_2 @ M_2 + M_2 @ A_2
        A_4 = A_2 @ A_2
        M_6 = A_4 @ M_2 + M_4 @ A_2
        A_6 = A_4 @ A_2
        W_1 = b[13] * A_6 + b[11] * A_4 + b[9] * A_2
        W_2 = b[7] * A_6 + b[5] * A_4 + b[3] * A_2 + b[1] * I
        W = A_6 @ W_1 + W_2
        Z_1 = b[12] * A_6 + b[10] * A_4 + b[8] * A_2
        Z_2 = b[6] * A_6 + b[4] * A_4 + b[2] * A_2 + b[0] * I
        U = A @ W
        V = A_6 @ Z_1 + Z_2
        L_W1 = b[13] * M_6 + b[11] * M_4 + b[9] * M_2
        L_W2 = b[7] * M_6 + b[5] * M_4 + b[3] * M_2
        L_Z1 = b[12] * M_6 + b[10] * M_4 + b[8] * M_2
        L_Z2 = b[6] * M_6 + b[4] * M_4 + b[2] * M_2
        L_W = A_6 @ L_W1 + M_6 @ W_1 + L_W2
        L_U = A @ L_W + E @ W
        L_V = A_6 @ L_Z1 + M_6 @ Z_1 + L_Z2
    lu_decom = torch.lu(-U + V)
    exp_A = torch.lu_solve(U + V, *lu_decom)
    dexp_A = torch.lu_solve(L_U + L_V + (L_U - L_V) @ exp_A, *lu_decom)
    return exp_A, dexp_A


def _square(s, R, L=None):
    """The `squaring` part of the `scaling-and-squaring` algorithm.

    This works both for the forward as well as the derivative of
    the matrix exponential.
    """
    s_max = torch.max(s).int()
    if s_max > 0:
        I = _eye_like(R)
        if L is not None:
            O = torch.zeros_like(R)
        indices = [(1) for k in range(len(R.shape) - 1)]
    for i in range(s_max):
        mask = i >= s
        matrices_mask = mask.view(-1, *indices)
        temp_eye = torch.clone(R).masked_scatter(matrices_mask, I)
        if L is not None:
            temp_zeros = torch.clone(R).masked_scatter(matrices_mask, O)
            L = temp_eye @ L + temp_zeros @ L
        R = R @ temp_eye
        del temp_eye, mask
    if L is not None:
        return R, L
    else:
        return R


def _expm_frechet_scaling_squaring(A, E, adjoint=False):
    """Numerical Fréchet derivative of matrix exponentiation.

    """
    assert A.shape[-1] == A.shape[-2] and len(A.shape) in [2, 3]
    has_batch_dim = True if len(A.shape) == 3 else False
    if adjoint == True:
        A = torch.transpose(A, -1, -2)
    s, m = _compute_scales(A)
    if torch.max(s) > 0:
        indices = [(1) for k in range(len(A.shape) - 1)]
        scaling_factors = torch.pow(2, -s).view(-1, *indices)
        A = A * scaling_factors
        E = E * scaling_factors
    exp_A, dexp_A = _expm_frechet_pade(A, E, m)
    exp_A, dexp_A = _square(s, exp_A, dexp_A)
    return dexp_A


def _expm_pade(A, m=7):
    assert m in [3, 5, 7, 9, 13]
    if m == 3:
        b = [120.0, 60.0, 12.0, 1.0]
    elif m == 5:
        b = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0]
    elif m == 7:
        b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0]
    elif m == 9:
        b = [17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0]
    elif m == 13:
        b = [6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0]
    I = _eye_like(A)
    if m != 13:
        U = b[1] * I
        V = b[0] * I
        if m >= 3:
            A_2 = A @ A
            U = U + b[3] * A_2
            V = V + b[2] * A_2
        if m >= 5:
            A_4 = A_2 @ A_2
            U = U + b[5] * A_4
            V = V + b[4] * A_4
        if m >= 7:
            A_6 = A_4 @ A_2
            U = U + b[7] * A_6
            V = V + b[6] * A_6
        if m == 9:
            A_8 = A_4 @ A_4
            U = U + b[9] * A_8
            V = V + b[8] * A_8
        U = A @ U
    else:
        A_2 = A @ A
        A_4 = A_2 @ A_2
        A_6 = A_4 @ A_2
        W_1 = b[13] * A_6 + b[11] * A_4 + b[9] * A_2
        W_2 = b[7] * A_6 + b[5] * A_4 + b[3] * A_2 + b[1] * I
        W = A_6 @ W_1 + W_2
        Z_1 = b[12] * A_6 + b[10] * A_4 + b[8] * A_2
        Z_2 = b[6] * A_6 + b[4] * A_4 + b[2] * A_2 + b[0] * I
        U = A @ W
        V = A_6 @ Z_1 + Z_2
    del A_2
    if m >= 5:
        del A_4
    if m >= 7:
        del A_6
    if m == 9:
        del A_8
    R = torch.lu_solve(U + V, *torch.lu(-U + V))
    del U, V
    return R


def _expm_scaling_squaring(A):
    """Scaling-and-squaring algorithm for matrix eponentiation.

    This is based on the observation that exp(A) = exp(A/k)^k, where
    e.g. k=2^s. The exponential exp(A/(2^s)) is calculated by a diagonal
    Padé approximation, where s is chosen based on the 1-norm of A, such
    that certain approximation guarantees can be given. exp(A) is then
    calculated by repeated squaring via exp(A/(2^s))^(2^s). This function
    works both for (n,n)-tensors as well as batchwise for (m,n,n)-tensors.
    """
    assert A.shape[-1] == A.shape[-2] and len(A.shape) in [2, 3]
    has_batch_dim = True if len(A.shape) == 3 else False
    s, m = _compute_scales(A)
    if torch.max(s) > 0:
        indices = [(1) for k in range(len(A.shape) - 1)]
        A = A * torch.pow(2, -s).view(-1, *indices)
    exp_A = _expm_pade(A, m)
    exp_A = _square(s, exp_A)
    return exp_A


class expm(Function):
    """Computes the matrix exponential.

    """

    @staticmethod
    def forward(ctx, M):
        expm_M = _expm_scaling_squaring(M)
        ctx.save_for_backward(M)
        return expm_M

    @staticmethod
    def backward(ctx, grad_out):
        M = ctx.saved_tensors[0]
        dexpm = _expm_frechet_scaling_squaring(M, grad_out, adjoint=True)
        return dexpm


def __calculate_kernel_matrix_exp__(weight, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return expm.apply(skew_symmetric_matrix)


def eye_like(M, device=None, dtype=None):
    """Creates an identity matrix of the same shape as another matrix.

    For matrix M, the output is same shape as M, if M is a (n,n)-matrix.
    If M is a batch of m matrices (i.e. a (m,n,n)-tensor), create a batch of
    (n,n)-identity-matrices.

    Args:
        M (torch.Tensor) : A tensor of either shape (n,n) or (m,n,n), for
            which either an identity matrix or a batch of identity matrices
            of the same shape will be created.
        device (torch.device, optional) : The device on which the output
            will be placed. By default, it is placed on the same device
            as M.
        dtype (torch.dtype, optional) : The dtype of the output. By default,
            it is the same dtype as M.

    Returns:
        torch.Tensor : Identity matrix or batch of identity matrices.
    """
    assert len(M.shape) in [2, 3]
    assert M.shape[-1] == M.shape[-2]
    n = M.shape[-1]
    if device is None:
        device = M.device
    if dtype is None:
        dtype = M.dtype
    eye = torch.eye(M.shape[-1], device=device, dtype=dtype)
    if len(M.shape) == 2:
        return eye
    else:
        m = M.shape[0]
        return eye.view(-1, n, n).expand(m, -1, -1)


def householder_matrix(unit_vector):
    if unit_vector.shape[-1] != 1:
        if len(unit_vector.shape) == 1:
            return torch.ones_like(unit_vector)
        unit_vector = unit_vector.view(*tuple(unit_vector.shape), 1)
    transform = 2 * unit_vector @ torch.transpose(unit_vector, -1, -2)
    return eye_like(transform) - transform


def normalize_matrix_rows(matrix, eps=1e-06):
    norms = torch.sqrt(torch.sum(matrix ** 2, dim=-2, keepdim=True) + eps)
    return matrix / norms


def householder_transform(matrix, n_reflections=-1, eps=1e-06):
    """Implements a product of Householder transforms.

    """
    if n_reflections == -1:
        n_reflections = matrix.shape[-1]
    if n_reflections > matrix.shape[-1]:
        warn('n_reflections is set higher than the number of rows.')
        n_reflections = matrix.shape[-1]
    matrix = normalize_matrix_rows(matrix, eps)
    if n_reflections == 0:
        output = torch.eye(matrix.shape[-2], dtype=matrix.dtype, device=matrix.device)
        if len(matrix.shape) == 3:
            output = output.view(1, matrix.shape[1], matrix.shape[1])
            output = output.expand(matrix.shape[0], -1, -1)
    for i in range(n_reflections):
        unit_vector = matrix[..., i:i + 1]
        householder = householder_matrix(unit_vector)
        if i == 0:
            output = householder
        else:
            output = output @ householder
    return output


def __calculate_kernel_matrix_householder__(weight, **kwargs):
    n_reflections = kwargs.get('n_reflections', -1)
    eps = kwargs.get('eps', 1e-06)
    weight_cols = weight.shape[-1]
    weight = weight[..., n_reflections:]
    return householder_transform(weight, n_reflections, eps)


def __initialize_weight__(kernel_matrix_shape: 'Tuple[int, ...]', stride: 'Tuple[int, ...]', method: 'str'='cayley', init: 'str'='haar', dtype: 'str'='float32', *args, **kwargs):
    """Function which computes specific orthogonal matrices.

    For some chosen method of parametrizing orthogonal matrices, this
    function outputs the required weights necessary to represent a
    chosen initialization as a Pytorch tensor of matrices.

    Args:
        kernel_matrix_shape : The output shape of the
            orthogonal matrices. Should be (num_matrices, height, width).
        stride : The stride for the invertible up- or
            downsampling for which this matrix is to be used. The length
            of ``stride`` should match the dimensionality of the data.
        method : The method for parametrising orthogonal matrices.
            Should be 'exp' or 'cayley'
        init : The matrix which should be represented. Should be
            'squeeze', 'pixel_shuffle', 'haar' or 'random'. 'haar' is only
            possible if ``stride`` is only 2.
        dtype : Numpy dtype which should be used for the matrix.
        *args: Variable length argument iterable.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor : Orthogonal matrices of shape ``kernel_matrix_shape``.
    """
    dim = len(stride)
    num_matrices = kernel_matrix_shape[0]
    assert method in ['exp', 'cayley', 'householder']
    if method == 'householder':
        warn('Householder parametrization not fully implemented yet. Only random initialization currently working.')
        init = 'random'
    if init == 'random':
        return torch.randn(kernel_matrix_shape)
    if init == 'haar' and set(stride) != {2}:
        None
        None
        init = 'squeeze'
    if init == 'haar' and set(stride) == {2}:
        if method == 'exp':
            p = np.pi / 4
            if dim == 1:
                weight = np.array([[[0, p], [0, 0]]], dtype=dtype)
            if dim == 2:
                weight = np.array([[[0, 0, p, p], [0, 0, -p, -p], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=dtype)
            if dim == 3:
                weight = np.array([[[0, p, p, 0, p, 0, 0, 0], [0, 0, 0, p, 0, p, 0, 0], [0, 0, 0, p, 0, 0, p, 0], [0, 0, 0, 0, 0, 0, 0, p], [0, 0, 0, 0, 0, p, p, 0], [0, 0, 0, 0, 0, 0, 0, p], [0, 0, 0, 0, 0, 0, 0, p], [0, 0, 0, 0, 0, 0, 0, 0]]], dtype=dtype)
            return torch.tensor(weight).repeat(num_matrices, 1, 1)
        elif method == 'cayley':
            if dim == 1:
                p = -np.sqrt(2) / (2 - np.sqrt(2))
                weight = np.array([[[0, p], [0, 0]]], dtype=dtype)
            if dim == 2:
                p = 0.5
                weight = np.array([[[0, 0, p, p], [0, 0, -p, -p], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=dtype)
            if dim == 3:
                p = 1 / np.sqrt(2)
                weight = np.array([[[0, -p, -p, 0, -p, 0, 0, 1 - p], [0, 0, 0, -p, 0, -p, p - 1, 0], [0, 0, 0, -p, 0, p - 1, -p, 0], [0, 0, 0, 0, 1 - p, 0, 0, -p], [0, 0, 0, 0, 0, -p, -p, 0], [0, 0, 0, 0, 0, 0, 0, -p], [0, 0, 0, 0, 0, 0, 0, -p], [0, 0, 0, 0, 0, 0, 0, 0]]], dtype=dtype)
            return torch.tensor(weight).repeat(num_matrices, 1, 1)
    if init in ['squeeze', 'pixel_shuffle', 'zeros']:
        if method == 'exp' or method == 'cayley':
            return torch.zeros(*kernel_matrix_shape)
    if type(init) is np.ndarray:
        init = torch.tensor(init.astype(dtype))
    if torch.is_tensor(init):
        if len(init.shape) == 2:
            init = init.reshape(1, *init.shape)
        if init.shape[0] == 1:
            init = init.repeat(num_matrices, 1, 1)
        assert init.shape == kernel_matrix_shape
        return init
    else:
        raise NotImplementedError('Unknown initialization.')


class OrthogonalResamplingLayer(torch.nn.Module):
    """Base class for orthogonal up- and downsampling operators.

    :param low_channel_number:
        Lower number of channels. These are the input
        channels in the case of downsampling ops, and the output
        channels in the case of upsampling ops.
    :param stride:
        The downsampling / upsampling factor for each dimension.
    :param channel_multiplier:
        The channel multiplier, i.e. the number
        by which the number of channels are multiplied (downsampling)
        or divided (upsampling).
    :param method:
        Which method to use for parametrizing orthogonal
        matrices which are used as convolutional kernels.
    """

    def __init__(self, low_channel_number: 'int', stride: 'Union[int, Tuple[int, ...]]', method: 'str'='cayley', init: 'Union[str, np.ndarray, torch.Tensor]'='haar', learnable: 'bool'=True, init_kwargs: 'dict'=None, **kwargs):
        super(OrthogonalResamplingLayer, self).__init__()
        self.low_channel_number = low_channel_number
        self.method = method
        self.stride = stride
        self.channel_multiplier = int(np.prod(stride))
        self.high_channel_number = self.channel_multiplier * low_channel_number
        if init_kwargs is None:
            init_kwargs = {}
        self.init_kwargs = init_kwargs
        self.kwargs = kwargs
        assert method in ['exp', 'cayley', 'householder']
        if method == 'exp':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_exp__
        elif method == 'cayley':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_cayley__
        elif method == 'householder':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_householder__
        self._kernel_matrix_shape = (self.low_channel_number,) + (self.channel_multiplier,) * 2
        self._kernel_shape = (self.high_channel_number, 1) + self.stride
        self.weight = torch.nn.Parameter(__initialize_weight__(kernel_matrix_shape=self._kernel_matrix_shape, stride=self.stride, method=self.method, init=init, **self.init_kwargs))
        self.weight.requires_grad = learnable

    @property
    def kernel_matrix(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return self.__calculate_kernel_matrix__(self.weight, **self.kwargs)

    @property
    def kernel(self):
        """The kernel associated with the invertible up-/downsampling.
        """
        return self.kernel_matrix.reshape(*self._kernel_shape)


class InvertibleDownsampling1D(OrthogonalResamplingLayer):

    def __init__(self, in_channels: 'int', stride: '_size_1_t'=2, method: 'str'='cayley', init: 'str'='haar', learnable: 'bool'=True, *args, **kwargs):
        stride = tuple(_single(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling1D, self).__init__(*args, low_channel_number=self.in_channels, stride=stride, method=method, init=init, learnable=learnable, **kwargs)

    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)

    def inverse(self, x):
        return F.conv_transpose1d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)


class InvertibleUpsampling1D(OrthogonalResamplingLayer):

    def __init__(self, in_channels: 'int', stride: '_size_1_t'=2, method: 'str'='cayley', init: 'str'='haar', learnable: 'bool'=True, *args, **kwargs):
        stride = tuple(_pair(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling1D, self).__init__(*args, low_channel_number=self.out_channels, stride=stride, method=method, init=init, learnable=learnable, **kwargs)

    def forward(self, x):
        return F.conv_transpose1d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)

    def inverse(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)


class InvertibleDownsampling2D(OrthogonalResamplingLayer):

    def __init__(self, in_channels: 'int', stride: '_size_2_t'=2, method: 'str'='cayley', init: 'str'='haar', learnable: 'bool'=True, *args, **kwargs):
        stride = tuple(_pair(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling2D, self).__init__(*args, low_channel_number=self.in_channels, stride=stride, method=method, init=init, learnable=learnable, **kwargs)

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)

    def inverse(self, x):
        return F.conv_transpose2d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)


class InvertibleUpsampling2D(OrthogonalResamplingLayer):

    def __init__(self, in_channels: 'int', stride: '_size_2_t'=2, method: 'str'='cayley', init: 'str'='haar', learnable: 'bool'=True, *args, **kwargs):
        stride = tuple(_pair(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling2D, self).__init__(*args, low_channel_number=self.out_channels, stride=stride, method=method, init=init, learnable=learnable, **kwargs)

    def forward(self, x):
        return F.conv_transpose2d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)

    def inverse(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)


class InvertibleDownsampling3D(OrthogonalResamplingLayer):

    def __init__(self, in_channels: 'int', stride: '_size_3_t'=2, method: 'str'='cayley', init: 'str'='haar', learnable: 'bool'=True, *args, **kwargs):
        stride = tuple(_triple(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling3D, self).__init__(*args, low_channel_number=self.in_channels, stride=stride, method=method, init=init, learnable=learnable, **kwargs)

    def forward(self, x):
        return F.conv3d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)

    def inverse(self, x):
        return F.conv_transpose3d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)


class InvertibleUpsampling3D(OrthogonalResamplingLayer):

    def __init__(self, in_channels: 'int', stride: '_size_3_t'=2, method: 'str'='cayley', init: 'str'='haar', learnable: 'bool'=True, *args, **kwargs):
        stride = tuple(_triple(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling3D, self).__init__(*args, low_channel_number=self.out_channels, stride=stride, method=method, init=init, learnable=learnable, **kwargs)

    def forward(self, x):
        return F.conv_transpose3d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)

    def inverse(self, x):
        return F.conv3d(x, self.kernel, stride=self.stride, groups=self.low_channel_number)


class SplitChannels(torch.nn.Module):

    def __init__(self, split_location):
        super(SplitChannels, self).__init__()
        self.split_location = split_location

    def forward(self, x):
        a, b = x[:, :self.split_location], x[:, self.split_location:]
        a, b = a.clone(), b.clone()
        del x
        return a, b

    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)


class ConcatenateChannels(torch.nn.Module):

    def __init__(self, split_location):
        super(ConcatenateChannels, self).__init__()
        self.split_location = split_location

    def forward(self, x, y):
        return torch.cat([x, y], dim=1)

    def inverse(self, x):
        a, b = x[:, :self.split_location], x[:, self.split_location:]
        a, b = a.clone(), b.clone()
        del x
        return a, b


class AdditiveCoupling(nn.Module):
    """Additive coupling layer, a basic invertible layer.

    By splitting the input activation :math:`x` and output activation :math:`y`
    into two groups of channels (i.e. :math:`(x_1, x_2) \\cong x` and
    :math:`(y_1, y_2) \\cong y`), `additive coupling layers` define an invertible
    mapping :math:`x \\mapsto y` via

    .. math::

       y_1 &= x_2

       y_2 &= x_1 + F(x_2),

    where the `coupling function` :math:`F` is an (almost) arbitrary mapping.
    :math:`F` just has to map from the space of :math:`x_2` to the space of
    :math:`x_1`. In practice, this can for instance be a sequence of
    convolutional layers with batch normalization.

    The inverse of the above mapping is computed algebraically via

    .. math::

       x_1 &= y_2 - F(y_1)

       x_2 &= y_1.

    *Warning*: Note that this is different from the definition of additive
    coupling layers in ``MemCNN``. Those are equivalent to two consecutive
    instances of the above-defined additive coupling layers. Hence, the
    variant implemented here is twice as memory-efficient as the variant from
    ``MemCNN``.

    :param F:
        The coupling function of the additive coupling layer, typically a
        sequence of neural network layers.
    :param channel_split_pos:
        The index of the channel at which the input and output activations are
        split.

    """

    def __init__(self, F: 'nn.Module', channel_split_pos: 'int'):
        super(AdditiveCoupling, self).__init__()
        self.F = F
        self.channel_split_pos = channel_split_pos

    def forward(self, x):
        x1, x2 = x[:, :self.channel_split_pos], x[:, self.channel_split_pos:]
        x1, x2 = x1.contiguous(), x2.contiguous()
        y1 = x2
        y2 = x1 + self.F.forward(x2)
        out = torch.cat([y1, y2], dim=1)
        return out

    def inverse(self, y):
        inverse_channel_split_pos = y.shape[1] - self.channel_split_pos
        y1, y2 = y[:, :inverse_channel_split_pos], y[:, inverse_channel_split_pos:]
        y1, y2 = y1.contiguous(), y2.contiguous()
        x2 = y1
        x1 = y2 - self.F.forward(y1)
        x = torch.cat([x1, x2], dim=1)
        return x


class OrthogonalChannelMixing(nn.Module):
    """Base class for all orthogonal channel mixing layers.

    """

    def __init__(self, in_channels: 'int', method: 'str'='cayley', learnable: 'bool'=True, **kwargs):
        super(OrthogonalChannelMixing, self).__init__()
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.zeros((in_channels, in_channels)), requires_grad=learnable)
        assert method in ['exp', 'cayley', 'householder']
        if method == 'exp':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_exp__
        elif method == 'cayley':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_cayley__
        elif method == 'householder':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_householder__
        self.kwargs = kwargs

    @property
    def kernel_matrix(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return self.__calculate_kernel_matrix__(self.weight, **self.kwargs)

    @property
    def kernel_matrix_transposed(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return torch.transpose(self.kernel_matrix, -1, -2)


class InvertibleChannelMixing1D(OrthogonalChannelMixing):
    """Orthogonal (and hence invertible) channel mixing layer for 1D data.

    This layer linearly combines the input channels to each output channel.
    Here, the number of output channels is the same as the number of input
    channels, and the matrix specifying the connectivity between the channels
    is orthogonal.

    :param in_channels:
        The number of input (and output) channels.
    :param method:
        The chosen method for parametrizing the orthogonal matrix which
        determines the orthogonal channel mixing. Either ``"exp"``, ``"cayley"``
        or ``"householder"``.

    """

    def __init__(self, in_channels: 'int', method: 'str'='cayley', learnable: 'bool'=True, **kwargs):
        super(InvertibleChannelMixing1D, self).__init__(in_channels=in_channels, method=method, learnable=learnable, **kwargs)
        self.kwargs = kwargs

    @property
    def kernel(self):
        return self.kernel_matrix.view(self.in_channels, self.in_channels, 1)

    def forward(self, x):
        return nn.functional.conv1d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose1d(x, self.kernel)


class InvertibleChannelMixing2D(OrthogonalChannelMixing):

    def __init__(self, in_channels: 'int', method: 'str'='cayley', learnable: 'bool'=True, **kwargs):
        super(InvertibleChannelMixing2D, self).__init__(in_channels=in_channels, method=method, learnable=learnable, **kwargs)
        self.kwargs = kwargs

    @property
    def kernel(self):
        return self.kernel_matrix.view(self.in_channels, self.in_channels, 1, 1)

    def forward(self, x):
        return nn.functional.conv2d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose2d(x, self.kernel)


class InvertibleChannelMixing3D(OrthogonalChannelMixing):

    def __init__(self, in_channels: 'int', method: 'str'='cayley', learnable: 'bool'=True, **kwargs):
        super(InvertibleChannelMixing3D, self).__init__(in_channels=in_channels, method=method, learnable=learnable, **kwargs)
        self.kwargs = kwargs

    @property
    def kernel(self):
        return self.kernel_matrix.view(self.in_channels, self.in_channels, 1, 1, 1)

    def forward(self, x):
        return nn.functional.conv3d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose3d(x, self.kernel)


def create_standard_module(in_channels, **kwargs):
    dim = kwargs.pop('dim', 2)
    depth = kwargs.pop('depth', 2)
    num_channels = get_num_channels(in_channels)
    num_F_in_channels = num_channels // 2
    num_F_out_channels = num_channels - num_F_in_channels
    module_index = kwargs.pop('module_index', 0)
    if np.mod(module_index, 2) == 0:
        num_F_in_channels, num_F_out_channels = num_F_out_channels, num_F_in_channels
    return AdditiveCoupling(F=StandardBlock(dim, num_F_in_channels, num_F_out_channels, depth=depth, **kwargs), channel_split_pos=num_F_out_channels)


def print_iunet_layout(iunet):
    left = []
    right = []
    splits = []
    middle_padding = [''] * iunet.num_levels
    output = [''] * iunet.num_levels
    for i in range(iunet.num_levels):
        left.append('-'.join([str(iunet.channels[i])] * iunet.architecture[i]))
        if i < iunet.num_levels - 1:
            splits.append('({}/{})'.format(iunet.skipped_channels[i], iunet.channels_before_downsampling[i]))
        else:
            splits.append('')
        right.append(splits[-1] + '-' + left[-1])
        left[-1] = left[-1] + '-' + splits[-1]
    for i in range(iunet.num_levels - 1, -1, -1):
        if i < iunet.num_levels - 1:
            middle_padding[i] = ''.join(['-'] * max([len(output[i + 1]) - len(splits[i]), 4]))
        output[i] = left[i] + middle_padding[i] + right[i]
    for i in range(iunet.num_levels):
        if i > 0:
            outside_padding = len(output[0]) - len(output[i])
            _left = outside_padding // 2
            left_padding = ''.join(['-'] * _left)
            _right = outside_padding - _left
            right_padding = ''.join(['-'] * _right)
            output[i] = ''.join([left_padding, output[i], right_padding])
        None


class iUNet(nn.Module):
    """Fully-invertible U-Net (iUNet).

    This model can be used for memory-efficient backpropagation, e.g. in
    high-dimensional (such as 3D) segmentation tasks.

    :param channels:
        The number of channels at each resolution. For example: If one wants
        5 resolution levels (i.e. 3 up-/downsampling operations), it should be
        a tuple of 4 numbers, e.g. ``(32,64,128,256,384)``.
    :param architecture:
        Determines the number of invertible layers at each
        resolution (both left and right), e.g. ``[2,3,4]`` results in the
        following structure::
            2-----2
             3---3
              4-4
        Must be the same length as ``channels``.
    :param dim: Either ``1``, ``2`` or ``3``, signifying whether a 1D, 2D or 3D
        invertible U-Net should be created.
    :param create_module_fn:
        Function which outputs an invertible layer. This layer
        should be a ``torch.nn.Module`` with a method ``forward(*x)``
        and a method ``inverse(*x)``. ``create_module_fn`` should have the
        signature ``create_module_fn(in_channels, **kwargs)``.
        Additional keyword arguments passed on via ``kwargs`` are
        ``dim`` (whether this is a 1D, 2D or 3D iUNet), the coordinates
        of the specific module within the iUNet (``branch``, ``level`` and
        ``module_index``) as well as ``architecture``. By default, this creates
        an additive coupling layer, whose block consists of a number of
        convolutional layers, followed by an instance normalization layer and
        a `leaky ReLU` activation function. The number of blocks can be
        controlled by setting ``"depth"`` in ``module_kwargs``, whose default
        value is ``2``.
    :param module_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to ``create_module_fn``.
    :param learnable_resampling:
        Whether to train the invertible learnable up- and downsampling
        or to leave it at the initialized values.
        Defaults to ``True``.
    :param resampling_stride:
        Controls the stride of the invertible up- and downsampling.
        The format can be either a single integer, a single tuple (where the
        length corresponds to the spatial dimensions of the data), or a list
        containing either of the last two options (where the length of the
        list has to be equal to the number of downsampling operations),
        For example: ``2`` would result in a up-/downsampling with a factor of 2
        along each dimension; ``(2,1,4)`` would apply (at every
        resampling) a factor of 2, 1 and 4 for the height, width and depth
        dimensions respectively, whereas for a 3D iUNet with 3 up-/downsampling
        stages, ``[(2,1,3), (2,2,2), (4,3,1)]`` would result in different
        strides at different up-/downsampling stages.
    :param resampling_method:
        Chooses the method for parametrizing orthogonal matrices for
        invertible up- and downsampling. Can be either ``"exp"`` (i.e.
        exponentiation of skew-symmetric matrices) or ``"cayley"`` (i.e.
        the Cayley transform, acting on skew-symmetric matrices).
        Defaults to ``"cayley"``.
    :param resampling_init:
        Sets the initialization for the learnable up- and downsampling
        operators. Can be ``"haar"``, ``"pixel_shuffle"`` (aliases:
        ``"squeeze"``, ``"zeros"``), a specific ``torch.Tensor`` or a
        ``numpy.ndarray``.
        Defaults to ``"haar"``, i.e. the `Haar transform`.
    :param resampling_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to the invertible up- and downsampling modules.
    :param channel_mixing_freq:
        How often an `invertible channel mixing` is applied, which is (in 2D)
        is an orthogonal 1x1-convolution. ``-1`` means that this will only be
        applied before the channel splitting and before the recombination in the
        decoder branch. For any other ``n``, this means that every ``n``-th
        module is followed by an invertible channel mixing. In particular,``0``
        deactivates the usage of invertible channel mixing.
        Defaults to ``-1``.
    :param channel_mixing_method:
        How the orthogonal matrix for invertible channel mixing is parametrized.
        Same has ``resampling_method``.
        Defaults to ``"cayley"``.
    :param channel_mixing_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to the invertible channel mixing modules.
    :param padding_mode:
        If downsampling is not possible without residue
        (e.g. when halving spatial odd-valued resolutions), the
        input gets padded to allow for invertibility of the padded
        input. padding_mode takes the same keywords as
        ``torch.nn.functional.pad`` for ``mode``. If set to ``None``,
        this behavior is deactivated.
        Defaults to ``"constant"``.
    :param padding_value:
        If ``padding_mode`` is set to `constant`, this
        is the value that the input is padded with, e.g. 0.
        Defaults to ``0``.
    :param revert_input_padding:
        Whether to revert the input padding in the output, such that the
        input resolution is preserved, even when padding is required.
        Defaults to ``True``.
    :param disable_custom_gradient:
        If set to ``True``, `normal backpropagation` (i.e. storing
        activations instead of reconstructing activations) is used.
        Defaults to ``False``.
    :param verbose:
        Level of verbosity. Currently only 0 (no warnings) or 1,
        which includes warnings.
        Defaults to ``1``.
    """

    def __init__(self, channels: 'Tuple[int, ...]', architecture: 'Tuple[int, ...]', dim: 'int', create_module_fn: 'CreateModuleFnType'=create_standard_module, module_kwargs: 'dict'=None, learnable_resampling: 'bool'=True, resampling_stride: 'int'=2, resampling_method: 'str'='cayley', resampling_init: 'Union[str, np.ndarray, torch.Tensor]'='haar', resampling_kwargs: 'dict'=None, learnable_channel_mixing: 'bool'=True, channel_mixing_freq: 'int'=-1, channel_mixing_method: 'str'='cayley', channel_mixing_kwargs: 'dict'=None, padding_mode: 'Union[str, type(None)]'='constant', padding_value: 'int'=0, revert_input_padding: 'bool'=True, disable_custom_gradient: 'bool'=False, verbose: 'int'=1, **kwargs: Any):
        super(iUNet, self).__init__()
        self.architecture = architecture
        self.dim = dim
        self.create_module_fn = create_module_fn
        self.disable_custom_gradient = disable_custom_gradient
        self.num_levels = len(architecture)
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = module_kwargs
        if len(channels) != len(architecture):
            raise AttributeError('channels must have the same length as architecture.')
        self.channels = [channels[0]]
        self.channels_before_downsampling = []
        self.skipped_channels = []
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.revert_input_padding = revert_input_padding
        self.resampling_stride = self.__format_stride__(resampling_stride)
        self.channel_multipliers = [int(np.prod(stride)) for stride in self.resampling_stride]
        self.resampling_method = resampling_method
        self.resampling_init = resampling_init
        if resampling_kwargs is None:
            resampling_kwargs = {}
        self.resampling_kwargs = resampling_kwargs
        self.downsampling_factors = self.__total_downsampling_factor__(self.resampling_stride)
        self.learnable_channel_mixing = learnable_channel_mixing
        self.channel_mixing_freq = channel_mixing_freq
        self.channel_mixing_method = channel_mixing_method
        if channel_mixing_kwargs is None:
            channel_mixing_kwargs = {}
        self.channel_mixing_kwargs = channel_mixing_kwargs
        desired_channels = channels
        channel_errors = []
        for i in range(len(architecture) - 1):
            factor = desired_channels[i + 1] / self.channels[i]
            skip_fraction = (self.channel_multipliers[i] - factor) / self.channel_multipliers[i]
            self.skipped_channels.append(int(max([1, np.round(self.channels[i] * skip_fraction)])))
            self.channels_before_downsampling.append(self.channels[i] - self.skipped_channels[-1])
            self.channels.append(self.channel_multipliers[i] * self.channels_before_downsampling[i])
            channel_errors.append(abs(self.channels[i] - desired_channels[i]) / desired_channels[i])
        self.channels = tuple(self.channels)
        self.channels_before_downsampling = tuple(self.channels_before_downsampling)
        self.skipped_channels = tuple(self.skipped_channels)
        if list(channels) != list(self.channels):
            None
        self.verbose = verbose
        downsampling_op = [InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D][dim - 1]
        upsampling_op = [InvertibleUpsampling1D, InvertibleUpsampling2D, InvertibleUpsampling3D][dim - 1]
        channel_mixing_op = [InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D][dim - 1]
        self.encoder_modules = nn.ModuleList()
        self.decoder_modules = nn.ModuleList()
        self.slice_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        for i, num_layers in enumerate(architecture):
            current_channels = self.channels[i]
            if i < len(architecture) - 1:
                self.slice_layers.append(InvertibleModuleWrapper(SplitChannels(self.skipped_channels[i]), disable=disable_custom_gradient))
                self.concat_layers.append(InvertibleModuleWrapper(ConcatenateChannels(self.skipped_channels[i]), disable=disable_custom_gradient))
                downsampling = downsampling_op(self.channels_before_downsampling[i], stride=self.resampling_stride[i], method=self.resampling_method, init=self.resampling_init, learnable=learnable_resampling, **resampling_kwargs)
                upsampling = upsampling_op(self.channels[i + 1], stride=self.resampling_stride[i], method=self.resampling_method, init=self.resampling_init, learnable=learnable_resampling, **resampling_kwargs)
                if learnable_resampling:
                    upsampling.kernel_matrix.data = downsampling.kernel_matrix.data
                self.downsampling_layers.append(InvertibleModuleWrapper(downsampling, disable=learnable_resampling))
                self.upsampling_layers.append(InvertibleModuleWrapper(upsampling, disable=learnable_resampling))
            self.encoder_modules.append(nn.ModuleList())
            self.decoder_modules.append(nn.ModuleList())

            def add_channel_mixing(obj, module_list):
                module_list.append(InvertibleModuleWrapper(channel_mixing_op(in_channels=obj.channels[i], method=obj.channel_mixing_method, learnable=obj.learnable_channel_mixing, **obj.channel_mixing_kwargs), disable=disable_custom_gradient))
            for j in range(num_layers):
                coordinate_kwargs = {'dim': self.dim, 'branch': 'encoder', 'level': i, 'module_index': j, 'architecture': self.architecture}
                self.encoder_modules[i].append(InvertibleModuleWrapper(create_module_fn(self.channels[i], **coordinate_kwargs, **module_kwargs), disable=disable_custom_gradient))
                coordinate_kwargs['branch'] = 'decoder'
                self.decoder_modules[i].append(InvertibleModuleWrapper(create_module_fn(self.channels[i], **coordinate_kwargs, **module_kwargs), disable=disable_custom_gradient))
                if self.channel_mixing_freq == -1 and i != len(architecture) - 1:
                    if j == 0:
                        add_channel_mixing(self, self.decoder_modules[i])
                    if j == num_layers - 1:
                        add_channel_mixing(self, self.encoder_modules[i])
                modulo = np.mod(j, self.channel_mixing_freq)
                if self.channel_mixing_freq > 0 and modulo == self.channel_mixing_freq - 1:
                    add_channel_mixing(self, self.encoder_modules[i])
                    add_channel_mixing(self, self.decoder_modules[i])

    def get_padding(self, x: 'torch.Tensor'):
        """Calculates the required padding for the input.

        """
        shape = x.shape[2:]
        factors = self.downsampling_factors
        padded_shape = [(int(np.ceil(s / f)) * f) for s, f in zip(shape, factors)]
        total_padding = [(p - s) for s, p in zip(shape, padded_shape)]
        padding = [None] * (2 * len(shape))
        padding[::2] = [(p - p // 2) for p in total_padding]
        padding[1::2] = [(p // 2) for p in total_padding]
        padding = padding[::-1]
        return padded_shape, padding

    def revert_padding(self, x: 'torch.Tensor', padding: 'List[int]'):
        """Reverses a given padding.
        
        :param x:
            The image that was originally padded.
        :param padding:
            The padding that is removed from ``x``.
        """
        shape = x.shape
        if self.dim == 1:
            x = x[:, :, padding[0]:shape[2] - padding[1]]
        if self.dim == 2:
            x = x[:, :, padding[2]:shape[2] - padding[3], padding[0]:shape[3] - padding[1]]
        if self.dim == 3:
            x = x[:, :, padding[4]:shape[2] - padding[5], padding[2]:shape[3] - padding[3], padding[0]:shape[4] - padding[1]]
        return x

    def __check_stride_format__(self, stride):
        """Check whether the stride has the correct format to be parsed.

        The format can be either a single integer, a single tuple (where the
        length corresponds to the spatial dimensions of the data), or a list
        containing either of the last two options (where the length of the
        list has to be equal to the number of downsampling operations),
        e.g. ``2`, ``(2,1,3)``, ``[(2,1,3), (2,2,2), (4,3,1)]``.
        """

        def raise_format_error():
            raise AttributeError('resampling_stride has the wrong format. The format can be either a single integer, a single tuple (where the length corresponds to the spatial dimensions of the data), or a list containing either of the last two options (where the length of the list has to be equal to the number of downsampling operations), e.g. 2, (2,1,3), [(2,1,3), (2,2,2), (4,3,1)]. ')
        if isinstance(stride, int):
            pass
        elif isinstance(stride, tuple):
            if len(stride) == self.dim:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_format_error()
        elif isinstance(stride, list):
            if len(stride) == self.num_levels - 1:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_format_error()
        else:
            raise_format_error()

    def __format_stride__(self, stride):
        """Parses the resampling_stride and reformats it into a standard format.
        """
        self.__check_stride_format__(stride)
        if isinstance(stride, int):
            return [(stride,) * self.dim] * (self.num_levels - 1)
        if isinstance(stride, tuple):
            return [stride] * (self.num_levels - 1)
        if isinstance(stride, list):
            for i, element in enumerate(stride):
                if isinstance(element, int):
                    stride[i] = (element,) * self.dim
            return stride

    def __total_downsampling_factor__(self, stride):
        """Calculates the total downsampling factor per spatial dimension.
        """
        factors = [1] * len(stride[0])
        for i, element_tuple in enumerate(stride):
            for j, element_int in enumerate(stride[i]):
                factors[j] = factors[j] * element_int
        return tuple(factors)

    def pad(self, x, padded_shape=None, padding=None):
        """Applies the chosen padding to the input, if required.
        """
        if self.padding_mode is None:
            raise AttributeError('padding_mode in {} is set to None.'.format(self))
        if padded_shape is None or padding is None:
            padded_shape, padding = self.get_padding(x)
        if padded_shape != x.shape[2:] and self.padding_mode is not None:
            if self.verbose:
                warnings.warn('Input resolution {} cannot be downsampled {}  times without residuals. Padding to resolution {} is  applied with mode {} to retain invertibility. Set padding_mode=None to deactivate padding. If so, expect errors.'.format(list(x.shape[2:]), len(self.architecture) - 1, padded_shape, self.padding_mode))
            x = nn.functional.pad(x, padding, self.padding_mode, self.padding_value)
        return x

    def encode(self, x, use_padding=False):
        """Encodes x, i.e. applies the contractive part of the iUNet.
        """
        codes = []
        if use_padding:
            x = self.pad(x)
        for i in range(self.num_levels):
            depth = len(self.encoder_modules[i])
            for j in range(depth):
                x = self.encoder_modules[i][j](x)
            if i < self.num_levels - 1:
                y, x = self.slice_layers[i](x)
                codes.append(y)
                x = self.downsampling_layers[i](x)
        if len(codes) == 0:
            return x
        codes.append(x)
        return tuple(codes)

    def decode(self, *codes):
        """Applies the expansive, i.e. decoding, portion of the iUNet.
        """
        if isinstance(codes, tuple):
            codes = list(codes)
        else:
            codes = [codes]
        x = codes.pop()
        for i in range(self.num_levels - 1, -1, -1):
            depth = len(self.decoder_modules[i])
            if i < self.num_levels - 1:
                y = codes.pop()
                x = self.upsampling_layers[i](x)
                x = self.concat_layers[i](y, x)
            for j in range(depth):
                x = self.decoder_modules[i][j](x)
        return x

    def forward(self, x: 'torch.Tensor'):
        """Applies the forward mapping of the iUNet to ``x``.
        """
        if not x.shape[1] == self.channels[0]:
            raise RuntimeError('The number of channels does not match in_channels.')
        if self.padding_mode is not None:
            padded_shape, padding = self.get_padding(x)
            x = self.pad(x, padded_shape, padding)
        code = self.encode(x, use_padding=False)
        x = self.decode(*code)
        if self.padding_mode is not None and self.revert_input_padding:
            x = self.revert_padding(x, padding)
        return x

    def decoder_inverse(self, x, use_padding=False):
        """Applies the inverse of the decoder portion of the iUNet.
        """
        codes = []
        if use_padding:
            x = self.pad(x)
        for i in range(self.num_levels):
            depth = len(self.decoder_modules[i])
            for j in range(depth - 1, -1, -1):
                x = self.decoder_modules[i][j].inverse(x)
            if i < self.num_levels - 1:
                y, x = self.concat_layers[i].inverse(x)
                codes.append(y)
                x = self.upsampling_layers[i].inverse(x)
        if len(codes) == 0:
            return x
        codes.append(x)
        return tuple(codes)

    def encoder_inverse(self, *codes):
        """Applies the inverse of the encoder portion of the iUNet.
        """
        if isinstance(codes, tuple):
            codes = list(codes)
        else:
            codes = [codes]
        x = codes.pop()
        for i in range(self.num_levels - 1, -1, -1):
            depth = len(self.encoder_modules[i])
            if i < self.num_levels - 1:
                y = codes.pop()
                x = self.downsampling_layers[i].inverse(x)
                x = self.slice_layers[i].inverse(y, x)
            for j in range(depth - 1, -1, -1):
                x = self.encoder_modules[i][j].inverse(x)
        return x

    def inverse(self, x: 'torch.Tensor'):
        """Applies the inverse of the iUNet to ``x``.
        """
        if not x.shape[1] == self.channels[0]:
            raise RuntimeError('The number of channels does not match in_channels.')
        if self.padding_mode is not None:
            padded_shape, padding = self.get_padding(x)
            x = self.pad(x, padded_shape, padding)
        code = self.decoder_inverse(x, use_padding=False)
        x = self.encoder_inverse(*code)
        if self.padding_mode is not None and self.revert_input_padding:
            if self.verbose:
                warnings.warn('revert_input_padding is set to True, which may yield non-exact reconstructions of the unpadded input.')
            x = self.revert_padding(x, padding)
        return x

    def print_layout(self):
        """Prints the layout of the iUNet.
        """
        print_iunet_layout(self)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConcatenateChannels,
     lambda: ([], {'split_location': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertibleChannelMixing1D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (InvertibleChannelMixing2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertibleChannelMixing3D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertibleDownsampling1D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (InvertibleDownsampling2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertibleDownsampling3D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertibleUpsampling2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SplitChannels,
     lambda: ([], {'split_location': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_cetmann_iunets(_paritybench_base):
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

