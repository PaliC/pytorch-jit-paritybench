import sys
_module = sys.modules[__name__]
del sys
Dataset = _module
Learner = _module
Dataset = _module
Model = _module
Module = _module
Opt = _module
main = _module
utils = _module
flops = _module
git_tools = _module
my_logger = _module
my_progress_bar = _module
my_rich_progress_bar = _module
my_save_config_callback = _module
utils_ = _module
Module = _module
Predict = _module
Simu = _module
Train = _module
utils = _module
Dataset = _module
FixedAarryIPDnet = _module
Module = _module
runIPDnetOff = _module
runIPDnetOn = _module
flops = _module
my_progress_bar = _module
my_rich_progress_bar = _module
utils_ = _module

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


import scipy


import scipy.io


import scipy.signal


import pandas


import random


import warnings


from copy import deepcopy


from collections import namedtuple


from torch.utils.data import Dataset


import matplotlib.pyplot as plt


from matplotlib import animation


import torch


import torch.optim as optim


from abc import ABC


from abc import abstractmethod


import math


import torch.nn as nn


import torch.nn.functional as F


from itertools import permutations


from scipy.optimize import linear_sum_assignment


from typing import Callable


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from torch import Tensor


from typing import Tuple


from typing import *


import time


import copy


from scipy.special import jn


from math import pi


from matplotlib import pyplot as plt


class FNblock(nn.Module):
    """
    The implementation of the full-band and narrow-band fusion block
    """

    def __init__(self, input_size, hidden_size=128, dropout=0.2, add_skip_dim=4, is_online=False, is_first=False):
        super(FNblock, self).__init__()
        self.input_size = input_size
        self.full_hidden_size = hidden_size // 2
        self.is_first = is_first
        self.is_online = is_online
        if self.is_online:
            self.narr_hidden_size = hidden_size
        else:
            self.narr_hidden_size = hidden_size // 2
        self.dropout = dropout
        self.dropout_full = nn.Dropout(p=self.dropout)
        self.dropout_narr = nn.Dropout(p=self.dropout)
        if is_first:
            self.fullLstm = nn.LSTM(input_size=self.input_size, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        else:
            self.fullLstm = nn.LSTM(input_size=self.input_size + add_skip_dim, hidden_size=self.full_hidden_size, batch_first=True, bidirectional=True)
        self.narrLstm = nn.LSTM(input_size=2 * self.full_hidden_size + add_skip_dim, hidden_size=self.narr_hidden_size, batch_first=True, bidirectional=not self.is_online)

    def forward(self, x, fb_skip, nb_skip):
        nb, nt, nf, nc = x.shape
        x = x.reshape(nb * nt, nf, -1)
        x, _ = self.fullLstm(x)
        x = self.dropout_full(x)
        x = torch.cat((x, fb_skip), dim=-1)
        x = x.view(nb, nt, nf, -1).permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)
        x, _ = self.narrLstm(x)
        x = self.dropout_narr(x)
        x = torch.cat((x, nb_skip), dim=-1)
        x = x.view(nb, nf, nt, -1).permute(0, 2, 1, 3)
        return x


class FN_SSL(nn.Module):
    """ 
    """

    def __init__(self, input_size=4, hidden_size=256, is_online=True, is_doa=False):
        """the block of full-band and narrow-band fusion
        """
        super(FN_SSL, self).__init__()
        self.is_online = is_online
        self.is_doa = is_doa
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = FNblock(input_size=self.input_size, is_online=self.is_online, is_first=True)
        self.block_2 = FNblock(input_size=self.hidden_size, is_online=self.is_online, is_first=False)
        self.block_3 = FNblock(input_size=self.hidden_size, is_online=self.is_online, is_first=False)
        self.emb2ipd = nn.Linear(256, 2)
        self.pooling = nn.AvgPool2d(kernel_size=(12, 1))
        self.tanh = nn.Tanh()
        if self.is_doa:
            self.ipd2doa = nn.Linear(512, 180)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        nb, nt, nf, nc = x.shape
        x, fb_skip, nb_skip = self.block_1(x)
        x, fb_skip, nb_skip = self.block_2(x, fb_skip=fb_skip, nb_skip=nb_skip)
        x, fb_skip, nb_skip = self.block_3(x, fb_skip=fb_skip, nb_skip=nb_skip)
        x = x.permute(0, 2, 1, 3).reshape(nb * nf, nt, -1)
        ipd = self.pooling(x)
        ipd = self.tanh(self.emb2ipd(ipd))
        _, nt2, _ = ipd.shape
        ipd = ipd.view(nb, nf, nt2, -1)
        ipd = ipd.permute(0, 2, 1, 3)
        ipd_real = ipd[:, :, :, 0]
        ipd_image = ipd[:, :, :, 1]
        result = torch.cat((ipd_real, ipd_image), dim=2)
        if self.is_doa:
            result = self.ipd2doa(result)
        return result


class STFT(nn.Module):
    """ Function: Get STFT coefficients of microphone signals (batch processing by pytorch)
        Args:       win_len         - the length of frame / window
                    win_shift_ratio - the ratio between frame shift and frame length
                    nfft            - the number of fft points
                    win             - window type 
                                    'boxcar': a rectangular window (equivalent to no window at all)
                                    'hann': a Hann window
					signal          - the microphone signals in time domain (nbatch, nsample, nch)
        Returns:    stft            - STFT coefficients (nbatch, nf, nt, nch)
    """

    def __init__(self, win_len, win_shift_ratio, nfft, win='hann'):
        super(STFT, self).__init__()
        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.win = win

    def forward(self, signal):
        nsample = signal.shape[-2]
        nch = signal.shape[-1]
        win_shift = int(self.win_len * self.win_shift_ratio)
        nf = int(self.nfft / 2) + 1
        nb = signal.shape[0]
        nt = np.floor((nsample - self.win_len) / win_shift + 1).astype(int)
        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64)
        if self.win == 'hann':
            window = torch.hann_window(window_length=self.win_len, device=signal.device)
        for ch_idx in range(0, nch, 1):
            stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len, window=window, center=False, normalized=False, return_complex=True)
        return stft


class ISTFT(nn.Module):
    """ Function: Get inverse STFT (batch processing by pytorch) 
		Args:		stft            - STFT coefficients (nbatch, nf, nt, nch)
					win_len         - the length of frame / window
					win_shift_ratio - the ratio between frame shift and frame length
					nfft            - the number of fft points
		Returns:	signal          - time-domain microphone signals (nbatch, nsample, nch)
	"""

    def __init__(self, win_len, win_shift_ratio, nfft):
        super(ISTFT, self).__init__()
        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft

    def forward(self, stft):
        nf = stft.shape[-3]
        nt = stft.shape[-2]
        nch = stft.shape[-1]
        nb = stft.shape[0]
        win_shift = int(self.win_len * self.win_shift_ratio)
        nsample = (nt - 1) * win_shift
        signal = torch.zeros((nb, nsample, nch))
        for ch_idx in range(0, nch, 1):
            signal_temp = torch.istft(stft[:, :, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, win_length=self.win_len, center=True, normalized=False, return_complex=False)
            signal[:, :, ch_idx] = signal_temp[:, 0:nsample]
        return signal


class getMetric(nn.Module):
    """  
	Call: 
	# single source 
	getmetric = at_module.getMetric(source_mode='single', metric_unfold=True)
	metric = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=['azi,'ele'], ae_TH=30, useVAD=False, vad_TH=vad_TH)
	# multiple source
	self.getmetric = getMetric(source_mode='multiple', metric_unfold=True)
	metric = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=['azi,'ele'], ae_TH=30, useVAD=False, vad_TH=[2/3, 0.2]])
	"""

    def __init__(self, source_mode='multiple', metric_unfold=True, large_number=10000, invalid_source_idx=10):
        """
		Args:
			useVAD	 	-  False, True
			soruce_mode	- 'single', 'multiple'
		"""
        super(getMetric, self).__init__()
        self.source_mode = source_mode
        self.metric_unfold = metric_unfold
        self.inf = large_number
        self.invlid_sidx = invalid_source_idx

    def forward(self, doa_gt, vad_gt, doa_est, vad_est, ae_mode, ae_TH=30, useVAD=True, vad_TH=[0.5, 0.5]):
        """
		Args:
			doa_gt, doa_est - (nb, nt, 2, ns) in degrees
			vad_gt, vad_est - (nb, nt, ns) binary values
			ae_mode 		- angle error mode, [*, *, *], * - 'azi', 'ele', 'aziele' 
			ae_TH			- angle error threshold, namely azimuth error threshold in degrees
			vad_TH 			- VAD threshold, [gtVAD_TH, estVAD_TH] 
		Returns:
			ACC, MAE or ACC, MD, FA, MAE, RMSE - [*, *, *]
		"""
        device = doa_gt.device
        if self.source_mode == 'single':
            nbatch, nt, naziele, nsources = doa_est.shape
            if useVAD == False:
                vad_gt = torch.ones((nbatch, nt, nsources))
                vad_est = torch.ones((nbatch, nt, nsources))
            else:
                vad_gt = vad_gt > vad_TH[0]
                vad_est = vad_est > vad_TH[1]
            vad_est = vad_est * vad_gt
            azi_error = self.angular_error(doa_est[:, :, 1, :], doa_gt[:, :, 1, :], 'azi')
            ele_error = self.angular_error(doa_est[:, :, 0, :], doa_gt[:, :, 0, :], 'ele')
            aziele_error = self.angular_error(doa_est.permute(2, 0, 1, 3), doa_gt.permute(2, 0, 1, 3), 'aziele')
            corr_flag = ((azi_error < ae_TH) + 0.0) * vad_est
            act_flag = 1 * vad_gt
            K_corr = torch.sum(corr_flag)
            ACC = torch.sum(corr_flag) / torch.sum(act_flag)
            MAE = []
            if 'ele' in ae_mode:
                MAE += [torch.sum(vad_gt * ele_error) / torch.sum(act_flag)]
            elif 'azi' in ae_mode:
                MAE += [torch.sum(vad_gt * azi_error) / torch.sum(act_flag)]
            elif 'aziele' in ae_mode:
                MAE += [torch.sum(vad_gt * aziele_error) / torch.sum(act_flag)]
            else:
                raise Exception('Angle error mode unrecognized')
            MAE = torch.tensor(MAE)
            metric = [ACC, MAE]
            if self.metric_unfold:
                metric = self.unfold_metric(metric)
            return metric
        elif self.source_mode == 'multiple':
            nbatch = doa_est.shape[0]
            nmode = len(ae_mode)
            acc = torch.zeros(nbatch, 1)
            md = torch.zeros(nbatch, 1)
            fa = torch.zeros(nbatch, 1)
            mae = torch.zeros(nbatch, nmode)
            rmse = torch.zeros(nbatch, nmode)
            for b_idx in range(nbatch):
                doa_gt_one = doa_gt[b_idx, ...]
                doa_est_one = doa_est[b_idx, ...]
                nt = doa_gt_one.shape[0]
                num_sources_gt = doa_gt_one.shape[2]
                num_sources_est = doa_est_one.shape[2]
                if useVAD == False:
                    vad_gt_one = torch.ones((nt, num_sources_gt))
                    vad_est_one = torch.ones((nt, num_sources_est))
                else:
                    vad_gt_one = vad_gt[b_idx, ...]
                    vad_est_one = vad_est[b_idx, ...]
                    vad_gt_one = vad_gt_one > vad_TH[0]
                    vad_est_one = vad_est_one > vad_TH[1]
                corr_flag = torch.zeros((nt, num_sources_gt))
                azi_error = torch.zeros((nt, num_sources_gt))
                ele_error = torch.zeros((nt, num_sources_gt))
                aziele_error = torch.zeros((nt, num_sources_gt))
                K_gt = vad_gt_one.sum(axis=1)
                vad_gt_sum = torch.reshape(vad_gt_one.sum(axis=1) > 0, (nt, 1)).repeat((1, num_sources_est))
                vad_est_one = vad_est_one * vad_gt_sum
                K_est = vad_est_one.sum(axis=1)
                for t_idx in range(nt):
                    num_gt = int(K_gt[t_idx].item())
                    num_est = int(K_est[t_idx].item())
                    if num_gt > 0 and num_est > 0:
                        est = doa_est_one[t_idx, :, vad_est_one[t_idx, :] > 0]
                        gt = doa_gt_one[t_idx, :, vad_gt_one[t_idx, :] > 0]
                        dist_mat_az = torch.zeros((num_gt, num_est))
                        dist_mat_el = torch.zeros((num_gt, num_est))
                        dist_mat_azel = torch.zeros((num_gt, num_est))
                        for gt_idx in range(num_gt):
                            for est_idx in range(num_est):
                                dist_mat_az[gt_idx, est_idx] = self.angular_error(est[1, est_idx], gt[1, gt_idx], 'azi')
                                dist_mat_el[gt_idx, est_idx] = self.angular_error(est[0, est_idx], gt[0, gt_idx], 'ele')
                                dist_mat_azel[gt_idx, est_idx] = self.angular_error(est[:, est_idx], gt[:, gt_idx], 'aziele')
                        invalid_assigns = dist_mat_az > ae_TH
                        dist_mat_az_bak = dist_mat_az.clone()
                        dist_mat_az_bak[invalid_assigns] = self.inf
                        assignment = list(linear_sum_assignment(dist_mat_az_bak))
                        assignment = self.judge_assignment(dist_mat_az_bak, assignment)
                        for src_idx in range(num_gt):
                            if assignment[src_idx] != self.invlid_sidx:
                                corr_flag[t_idx, src_idx] = 1
                                azi_error[t_idx, src_idx] = dist_mat_az[src_idx, assignment[src_idx]]
                                ele_error[t_idx, src_idx] = dist_mat_el[src_idx, assignment[src_idx]]
                                aziele_error[t_idx, src_idx] = dist_mat_azel[src_idx, assignment[src_idx]]
                K_corr = corr_flag.sum(axis=1)
                acc[b_idx, :] = K_corr.sum(axis=0) / K_gt.sum(axis=0)
                md[b_idx, :] = (K_gt.sum(axis=0) - K_corr.sum(axis=0)) / K_gt.sum(axis=0)
                fa[b_idx, :] = (K_est.sum(axis=0) - K_corr.sum(axis=0)) / K_gt.sum(axis=0)
                mae_temp = []
                rmse_temp = []
                if 'ele' in ae_mode:
                    mae_temp += [(ele_error * corr_flag).sum(axis=0).sum() / (K_corr.sum(axis=0) + 1e-05)]
                    rmse_temp += [torch.sqrt((ele_error * ele_error * corr_flag).sum(axis=0).sum() / (K_corr.sum(axis=0) + 1e-05))]
                elif 'azi' in ae_mode:
                    mae_temp += [(azi_error * corr_flag).sum(axis=0).sum() / (K_corr.sum(axis=0) + 1e-05)]
                    rmse_temp += [torch.sqrt((azi_error * azi_error * corr_flag).sum(axis=0).sum() / (K_corr.sum(axis=0) + 1e-05))]
                elif 'aziele' in ae_mode:
                    mae_temp += [(aziele_error * corr_flag).sum(axis=0).sum() / (K_corr.sum(axis=0) + 1e-05)]
                    rmse_temp += [torch.sqrt((aziele_error * aziele_error * corr_flag).sum(axis=0).sum() / (K_corr.sum(axis=0) + 1e-05))]
                else:
                    raise Exception('Angle error mode unrecognized')
                mae[b_idx, :] = torch.tensor(mae_temp)
                rmse[b_idx, :] = torch.tensor(rmse_temp)
            ACC = torch.mean(acc, dim=0)
            MD = torch.mean(md, dim=0)
            FA = torch.mean(fa, dim=0)
            MAE = torch.mean(mae, dim=0)
            RMSE = torch.mean(rmse, dim=0)
            metric = [ACC, MD, FA, MAE, RMSE]
            if self.metric_unfold:
                metric = self.unfold_metric(metric)
            return metric

    def judge_assignment(self, dist_mat, assignment):
        final_assignment = torch.tensor([self.invlid_sidx for i in range(dist_mat.shape[0])])
        for i in range(min(dist_mat.shape[0], dist_mat.shape[1])):
            if dist_mat[assignment[0][i], assignment[1][i]] != self.inf:
                final_assignment[assignment[0][i]] = assignment[1][i]
            else:
                final_assignment[i] = self.invlid_sidx
        return final_assignment

    def angular_error(self, est, gt, ae_mode):
        """
		Function: return angular error in degrees
		"""
        if ae_mode == 'azi':
            ae = torch.abs((est - gt + 180) % 360 - 180)
        elif ae_mode == 'ele':
            ae = torch.abs(est - gt)
        elif ae_mode == 'aziele':
            ele_gt = gt[0, ...].float() / 180 * np.pi
            azi_gt = gt[1, ...].float() / 180 * np.pi
            ele_est = est[0, ...].float() / 180 * np.pi
            azi_est = est[1, ...].float() / 180 * np.pi
            aux = torch.cos(ele_gt) * torch.cos(ele_est) + torch.sin(ele_gt) * torch.sin(ele_est) * torch.cos(azi_gt - azi_est)
            aux[aux.gt(0.99999)] = 0.99999
            aux[aux.lt(-0.99999)] = -0.99999
            ae = torch.abs(torch.acos(aux)) * 180 / np.pi
        else:
            raise Exception('Angle error mode unrecognized')
        return ae

    def unfold_metric(self, metric):
        metric_unfold = []
        for m in metric:
            if m.numel() != 1:
                for n in range(m.numel()):
                    metric_unfold += [m[n]]
            else:
                metric_unfold += [m]
        return metric_unfold


class visDOA(nn.Module):
    """ Function: Visualize localization results
	"""

    def __init__(self):
        super(visDOA, self).__init__()

    def forward(self, doa_gt, vad_gt, doa_est, vad_est, vad_TH, time_stamp, doa_invalid=200):
        """ Args:
				doa_gt, doa_est - (nt, 2, ns) in degrees
				vad_gt, vad_est - (nt, ns)  
				vad_TH 			- VAD threshold, [gtVAD_TH, estVAD_TH] 
			Returns: plt
		"""
        plt.switch_backend('agg')
        doa_mode = ['Elevation [º]', 'Azimuth [º]']
        range_mode = [[0, 180], [0, 180]]
        num_sources_gt = doa_gt.shape[-1]
        num_sources_pred = doa_est.shape[-1]
        ndoa_mode = 1
        for doa_mode_idx in [1]:
            valid_flag_all = np.sum(vad_gt, axis=-1) > 0
            valid_flag_all = valid_flag_all[:, np.newaxis, np.newaxis].repeat(doa_gt.shape[1], axis=1).repeat(doa_gt.shape[2], axis=2)
            valid_flag_gt = vad_gt > vad_TH[0]
            valid_flag_gt = valid_flag_gt[:, np.newaxis, :].repeat(doa_gt.shape[1], axis=1)
            doa_gt_v = np.where(valid_flag_gt, doa_gt, doa_invalid)
            doa_gt_silence_v = np.where(valid_flag_gt == 0, doa_gt, doa_invalid)
            valid_flag_pred = vad_est > vad_TH[1]
            valid_flag_pred = valid_flag_pred[:, np.newaxis, :].repeat(doa_est.shape[1], axis=1)
            doa_pred_v = np.where(valid_flag_pred & valid_flag_all, doa_est, doa_invalid)
            plt.subplot(ndoa_mode, 1, 1)
            plt.grid(linestyle=':', color='silver')
            for source_idx in range(num_sources_gt):
                plt_gt_silence = plt.scatter(time_stamp, doa_gt_silence_v[:, doa_mode_idx, source_idx], label='GT_silence', c='whitesmoke', marker='.', linewidth=1)
                plt_gt = plt.scatter(time_stamp, doa_gt_v[:, doa_mode_idx, source_idx], label='GT', c='lightgray', marker='o', linewidth=1.5)
            for source_idx in range(num_sources_pred):
                plt_est = plt.scatter(time_stamp, doa_pred_v[:, doa_mode_idx, source_idx], label='EST', c='firebrick', marker='.', linewidth=0.8)
            plt.gca().set_prop_cycle(None)
            plt.legend(handles=[plt_gt_silence, plt_gt, plt_est])
            plt.xlabel('Time [s]')
            plt.ylabel(doa_mode[doa_mode_idx])
            plt.ylim(range_mode[doa_mode_idx][0], range_mode[doa_mode_idx][1])
        return plt


class AddChToBatch(nn.Module):
    """ Change dimension from  (nb, nch, ...) to (nb*(nch-1), ...) 
	"""

    def __init__(self, ch_mode):
        super(AddChToBatch, self).__init__()
        self.ch_mode = ch_mode

    def forward(self, data):
        nb = data.shape[0]
        nch = data.shape[1]
        if self.ch_mode == 'M':
            data_adjust = torch.zeros((nb * (nch - 1), 2) + data.shape[2:], dtype=torch.complex64)
            for b_idx in range(nb):
                st = b_idx * (nch - 1)
                ed = (b_idx + 1) * (nch - 1)
                data_adjust[st:ed, 0, ...] = data[b_idx, 0:1, ...].expand((nch - 1,) + data.shape[2:])
                data_adjust[st:ed, 1, ...] = data[b_idx, 1:nch, ...]
        elif self.ch_mode == 'MM':
            data_adjust = torch.zeros((nb * int((nch - 1) * nch / 2), 2) + data.shape[2:], dtype=torch.complex64)
            for b_idx in range(nb):
                for ch_idx in range(nch - 1):
                    st = b_idx * int((nch - 1) * nch / 2) + int((2 * nch - 2 - ch_idx + 1) * ch_idx / 2)
                    ed = b_idx * int((nch - 1) * nch / 2) + int((2 * nch - 2 - ch_idx) * (ch_idx + 1) / 2)
                    data_adjust[st:ed, 0, ...] = data[b_idx, ch_idx:ch_idx + 1, ...].expand((nch - ch_idx - 1,) + data.shape[2:])
                    data_adjust[st:ed, 1, ...] = data[b_idx, ch_idx + 1:, ...]
        return data_adjust.contiguous()


class RemoveChFromBatch(nn.Module):
    """ Change dimension from (nb*nmic, nt, nf) to (nb, nmic, nt, nf)
	"""

    def __init__(self, ch_mode):
        super(RemoveChFromBatch, self).__init__()
        self.ch_mode = ch_mode

    def forward(self, data, nb):
        nmic = int(data.shape[0] / nb)
        data_adjust = torch.zeros((nb, nmic) + data.shape[1:], dtype=torch.float32)
        for b_idx in range(nb):
            st = b_idx * nmic
            ed = (b_idx + 1) * nmic
            data_adjust[b_idx, ...] = data[st:ed, ...]
        return data_adjust.contiguous()


class DPIPD(nn.Module):
    """ Complex-valued Direct-path inter-channel phase difference	
	"""

    def __init__(self, ndoa_candidate, mic_location, nf=257, fre_max=8000, ch_mode='M', speed=343.0, search_space_azi=[0, np.pi], search_space_ele=[np.pi / 2, np.pi / 2]):
        super(DPIPD, self).__init__()
        self.ndoa_candidate = ndoa_candidate
        self.mic_location = mic_location
        self.nf = nf
        self.fre_max = fre_max
        self.speed = speed
        self.ch_mode = ch_mode
        nmic = mic_location.shape[-2]
        nele = ndoa_candidate[0]
        nazi = ndoa_candidate[1]
        ele_candidate = np.linspace(search_space_ele[0], search_space_ele[1], nele)
        azi_candidate = np.linspace(search_space_azi[0], search_space_azi[1], nazi)
        ITD = np.empty((nele, nazi, nmic, nmic))
        IPD = np.empty((nele, nazi, nf, nmic, nmic))
        fre_range = np.linspace(0.0, fre_max, nf)
        for m1 in range(nmic):
            for m2 in range(nmic):
                r = np.stack([np.outer(np.sin(ele_candidate), np.cos(azi_candidate)), np.outer(np.sin(ele_candidate), np.sin(azi_candidate)), np.tile(np.cos(ele_candidate), [nazi, 1]).transpose()], axis=2)
                ITD[:, :, m1, m2] = np.dot(r, mic_location[m2, :] - mic_location[m1, :]) / speed
                IPD[:, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, :], [nele, nazi, 1]) * np.tile(ITD[:, :, np.newaxis, m1, m2], [1, 1, nf])
        dpipd_template_ori = np.exp(1.0j * IPD)
        self.dpipd_template = self.data_adjust(dpipd_template_ori)
        del ITD, IPD

    def forward(self, source_doa=None):
        mic_location = self.mic_location
        nf = self.nf
        fre_max = self.fre_max
        speed = self.speed
        if source_doa is not None:
            source_doa = source_doa.transpose(0, 1, 3, 2)
            nmic = mic_location.shape[-2]
            nb = source_doa.shape[0]
            nsource = source_doa.shape[-2]
            ntime = source_doa.shape[-3]
            ITD = np.empty((nb, ntime, nsource, nmic, nmic))
            IPD = np.empty((nb, ntime, nsource, nf, nmic, nmic))
            fre_range = np.linspace(0.0, fre_max, nf)
            for m1 in range(1):
                for m2 in range(1, nmic):
                    r = np.stack([np.sin(source_doa[:, :, :, 0]) * np.cos(source_doa[:, :, :, 1]), np.sin(source_doa[:, :, :, 0]) * np.sin(source_doa[:, :, :, 1]), np.cos(source_doa[:, :, :, 0])], axis=3)
                    ITD[:, :, :, m1, m2] = np.dot(r, mic_location[m1, :] - mic_location[m2, :]) / speed
                    IPD[:, :, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, np.newaxis, :], [nb, ntime, nsource, 1]) * np.tile(ITD[:, :, :, np.newaxis, m1, m2], [1, 1, 1, nf]) * -1
            dpipd_ori = np.exp(1.0j * IPD)
            dpipd = self.data_adjust(dpipd_ori)
            dpipd = dpipd.transpose(0, 1, 3, 4, 2)
        else:
            dpipd = None
        return self.dpipd_template, dpipd

    def data_adjust(self, data):
        if self.ch_mode == 'M':
            data_adjust = data[..., 0, 1:]
        elif self.ch_mode == 'MM':
            nmic = data.shape[-1]
            data_adjust = np.empty(data.shape[:-2] + (int(nmic * (nmic - 1) / 2),), dtype=np.complex64)
            for mic_idx in range(nmic - 1):
                st = int((2 * nmic - 2 - mic_idx + 1) * mic_idx / 2)
                ed = int((2 * nmic - 2 - mic_idx) * (mic_idx + 1) / 2)
                data_adjust[..., st:ed] = data[..., mic_idx, mic_idx + 1:]
        else:
            raise Exception('Microphone channel mode unrecognised')
        return data_adjust


class SourceDetectLocalize(nn.Module):

    def __init__(self, max_num_sources, source_num_mode='unkNum', meth_mode='IDL'):
        super(SourceDetectLocalize, self).__init__()
        self.max_num_sources = max_num_sources
        self.source_num_mode = source_num_mode
        self.meth_mode = meth_mode

    def forward(self, pred_ipd, dpipd_template, doa_candidate):
        device = pred_ipd.device
        pred_ipd = pred_ipd.detach()
        nb, nt, nf, nmic = pred_ipd.shape
        nele, nazi, _, _ = dpipd_template.shape
        dpipd_template = dpipd_template[np.newaxis, ...].repeat(nb, 1, 1, 1, 1)
        ele_candidate = doa_candidate[0]
        azi_candidate = doa_candidate[1]
        pred_ss = torch.bmm(pred_ipd.contiguous().view(nb, nt, -1), dpipd_template.contiguous().view(nb, nele, nazi, -1).permute(0, 3, 1, 2).view(nb, nmic * nf, -1)) / (nmic * nf / 2)
        pred_ss = pred_ss.view(nb, nt, nele, nazi)
        pred_DOAs = torch.zeros((nb, nt, 2, self.max_num_sources), dtype=torch.float32, requires_grad=False)
        pred_VADs = torch.zeros((nb, nt, self.max_num_sources), dtype=torch.float32, requires_grad=False)
        if self.meth_mode == 'IDL':
            for source_idx in range(self.max_num_sources):
                map = torch.bmm(pred_ipd.contiguous().view(nb, nt, -1), dpipd_template.contiguous().view(nb, nele, nazi, -1).permute(0, 3, 1, 2).view(nb, nmic * nf, -1)) / (nmic * nf / 2)
                map = map.view(nb, nt, nele, nazi)
                max_flat_idx = map.reshape((nb, nt, -1)).argmax(2)
                ele_max_idx, azi_max_idx = np.unravel_index(max_flat_idx.cpu().numpy(), map.shape[2:])
                pred_DOA = np.stack((ele_candidate[ele_max_idx], azi_candidate[azi_max_idx]), axis=-1)
                pred_DOA = torch.from_numpy(pred_DOA)
                pred_DOAs[:, :, :, source_idx] = pred_DOA
                max_dpipd_template = torch.zeros((nb, nt, nf, nmic), dtype=torch.float32, requires_grad=False)
                for b_idx in range(nb):
                    for t_idx in range(nt):
                        max_dpipd_template[b_idx, t_idx, :, :] = dpipd_template[b_idx, ele_max_idx[b_idx, t_idx], azi_max_idx[b_idx, t_idx], :, :] * 1.0
                        ratio = torch.sum(max_dpipd_template[b_idx, t_idx, :, :] * pred_ipd[b_idx, t_idx, :, :]) / torch.sum(max_dpipd_template[b_idx, t_idx, :, :] * max_dpipd_template[b_idx, t_idx, :, :])
                        max_dpipd_template[b_idx, t_idx, :, :] = ratio * max_dpipd_template[b_idx, t_idx, :, :]
                        if self.source_num_mode == 'kNum':
                            pred_VADs[b_idx, t_idx, source_idx] = 1
                        elif self.source_num_mode == 'unkNum':
                            pred_VADs[b_idx, t_idx, source_idx] = ratio * 1
                pred_ipd = pred_ipd - max_dpipd_template
        elif self.meth_mode == 'PD':
            ss = deepcopy(pred_ss[:, :, :, 0:-1])
            ss_top = torch.cat((ss[:, :, 0:1, :], ss[:, :, 0:-1, :]), dim=2)
            ss_bottom = torch.cat((ss[:, :, 1:, :], ss[:, :, -1:, :]), dim=2)
            ss_left = torch.cat((ss[:, :, :, -1:], ss[:, :, :, 0:-1]), dim=3)
            ss_right = torch.cat((ss[:, :, :, 1:], ss[:, :, :, 0:1]), dim=3)
            ss_top_left = torch.cat((torch.cat((ss[:, :, 0:1, -1:], ss[:, :, 0:1, 0:-1]), dim=3), torch.cat((ss[:, :, 0:-1, -1:], ss[:, :, 0:-1, 0:-1]), dim=3)), dim=2)
            ss_top_right = torch.cat((torch.cat((ss[:, :, 0:1, 1:], ss[:, :, 0:1, 0:1]), dim=3), torch.cat((ss[:, :, 0:-1, 1:], ss[:, :, 0:-1, 0:1]), dim=3)), dim=2)
            ss_bottom_left = torch.cat((torch.cat((ss[:, :, 1:, -1:], ss[:, :, 1:, 0:-1]), dim=3), torch.cat((ss[:, :, -1:, -1:], ss[:, :, -1:, 0:-1]), dim=3)), dim=2)
            ss_bottom_right = torch.cat((torch.cat((ss[:, :, 1:, 1:], ss[:, :, 1:, 0:1]), dim=3), torch.cat((ss[:, :, -1:, 1:], ss[:, :, -1:, 0:1]), dim=3)), dim=2)
            peaks = (ss > ss_top) & (ss > ss_bottom) & (ss > ss_left) & (ss > ss_right) & (ss > ss_top_left) & (ss > ss_top_right) & (ss > ss_bottom_left) & (ss > ss_bottom_right)
            peaks = torch.cat((peaks, torch.zeros_like(peaks[:, :, :, 0:1])), dim=3)
            peaks_reshape = peaks.reshape((nb, nt, -1))
            ss_reshape = pred_ss.reshape((nb, nt, -1))
            for b_idx in range(nb):
                for t_idx in range(nt):
                    peaks_idxs = torch.nonzero(peaks_reshape[b_idx, t_idx, :] == 1)
                    max_flat_idx = sorted(peaks_idxs, key=lambda k: ss_reshape[b_idx, t_idx, k], reverse=True)
                    max_flat_idx = max_flat_idx[0:self.max_num_sources]
                    max_flat_peakvalue = ss_reshape[b_idx, t_idx, max_flat_idx]
                    max_flat_idx = [i.cpu() for i in max_flat_idx]
                    ele_max_idx, azi_max_idx = np.unravel_index(max_flat_idx, peaks.shape[2:])
                    pred_DOA = np.stack((ele_candidate[ele_max_idx], azi_candidate[azi_max_idx]), axis=-1)
                    pred_DOA = torch.from_numpy(pred_DOA)
                    pred_DOAs[b_idx, t_idx, :, :] = pred_DOA.transpose(1, 0) * 1
                    if self.source_num_mode == 'kNum':
                        pred_VADs[b_idx, t_idx, :] = 1
                    elif self.source_num_mode == 'unkNum':
                        pred_VADs[b_idx, t_idx, :] = max_flat_peakvalue * 1
        else:
            raise Exception('Localizion method is unrecognized')
        track_enable = False
        if track_enable == True:
            for b_idx in range(nb):
                for t_idx in range(nt - 1):
                    temp = []
                    for source_idx in range(self.max_num_sources):
                        temp += [pred_DOAs[b_idx, t_idx + 1, :, source_idx]]
                    pair_permute = list(permutations(temp, self.max_num_sources))
                    diff = torch.zeros(len(pair_permute))
                    for pair_idx in range(len(pair_permute)):
                        pair = torch.stack(pair_permute[pair_idx]).permute(1, 0)
                        abs_diff1 = torch.abs(pair - pred_DOAs[b_idx, t_idx, :, :])
                        abs_diff2 = deepcopy(abs_diff1)
                        abs_diff2[1, :] = np.pi * 2 - abs_diff1[1, :]
                        abs_diff = torch.min(abs_diff1, abs_diff2)
                        diff[pair_idx] = torch.sum(abs_diff)
                    pair_idx_sim = torch.argmin(diff)
                    pred_DOAs[b_idx, t_idx + 1, :, :] = torch.stack(pair_permute[pair_idx_sim]).permute(1, 0)
        return pred_DOAs, pred_VADs, pred_ss


class PredDOA(nn.Module):

    def __init__(self, source_num_mode='UnkNum', max_num_sources=1, max_track=2, res_the=1, res_phi=180, fs=16000, nfft=512, ch_mode='M', dev='cuda', mic_location=None, is_linear_array=True, is_planar_array=True):
        super(PredDOA, self).__init__()
        self.nfft = nfft
        self.fre_max = fs / 2
        self.ch_mode = ch_mode
        self.source_num_mode = source_num_mode
        self.max_num_sources = max_num_sources
        self.fre_range_used = range(1, int(self.nfft / 2) + 1, 1)
        self.removebatch = RemoveChFromBatch(ch_mode=self.ch_mode)
        self.dev = dev
        self.max_track = max_track
        if is_linear_array:
            search_space = [0, np.pi]
        self.gerdpipd = DPIPD(ndoa_candidate=[res_the, res_phi], mic_location=mic_location, nf=int(self.nfft / 2) + 1, fre_max=self.fre_max, ch_mode=self.ch_mode, speed=340)
        self.getmetric = getMetric(source_mode='multiple', metric_unfold=True)

    def forward(self, pred_batch, gt_batch, idx):
        pred_batch, _ = self.pred2DOA(pred_batch=pred_batch, gt_batch=gt_batch)
        metric = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch, idx=idx)
        return metric

    def pred2DOA(self, pred_batch, gt_batch):
        """
		Convert Estimated IPD of mul-track to DOA
	    """
        nb, nt, ndoa, nmic, nmax = pred_batch.shape
        pred_ipd = pred_batch.permute(0, 3, 1, 2, 4).reshape(nb * nmic, nt, ndoa, nmax)
        return_pred_batch_doa = torch.zeros((nb, nt, 2, self.max_track))
        return_pred_batch_vad = torch.zeros((nb, nt, self.max_track))
        for i in range(self.max_track):
            pred_batch_temp, gt_batch_temp = self.pred2DOA_track(pred_ipd[:, :, :, i], gt_batch)
            return_pred_batch_doa[:, :, :, i:i + 1] = pred_batch_temp[0]
            return_pred_batch_vad[:, :, i:i + 1] = pred_batch_temp[1]
        pred_batch = [return_pred_batch_doa]
        pred_batch += [return_pred_batch_vad]
        pred_batch += [pred_ipd]
        if gt_batch is not None:
            if type(gt_batch) is list:
                for idx in range(len(gt_batch)):
                    gt_batch[idx] = gt_batch[idx].detach()
            else:
                gt_batch = gt_batch.detach()
        return pred_batch, gt_batch

    def pred2DOA_track(self, pred_batch=None, gt_batch=None, time_pool_size=None):
        """
		Convert Estimated IPD of one track to DOA
	    """
        if pred_batch is not None:
            pred_batch = pred_batch.detach()
            dpipd_template_sbatch, _ = self.gerdpipd()
            nele, nazi, _, nmic = dpipd_template_sbatch.shape
            nbnmic, nt, nf = pred_batch.shape
            nb = int(nbnmic / nmic)
            dpipd_template_sbatch = np.concatenate((dpipd_template_sbatch.real[:, :, self.fre_range_used, :], dpipd_template_sbatch.imag[:, :, self.fre_range_used, :]), axis=2).astype(np.float32)
            dpipd_template = np.tile(dpipd_template_sbatch[np.newaxis, :, :, :, :], [nb, 1, 1, 1, 1])
            dpipd_template = torch.from_numpy(dpipd_template)
            pred_rebatch = self.removebatch(pred_batch, nb).permute(0, 2, 3, 1)
            pred_rebatch = pred_rebatch
            dpipd_template = dpipd_template
            if time_pool_size is not None:
                nt_pool = int(nt / time_pool_size)
                pred_phases = torch.zeros((nb, nt_pool, nf, nmic), dtype=torch.float32, requires_grad=False)
                pred_phases = pred_phases
                for t_idx in range(nt_pool):
                    pred_phases[:, t_idx, :, :] = torch.mean(pred_rebatch[:, t_idx * time_pool_size:(t_idx + 1) * time_pool_size, :, :], dim=1)
                pred_rebatch = pred_phases * 1
                nt = nt_pool * 1
            pred_spatial_spectrum = torch.bmm(pred_rebatch.contiguous().view(nb, nt, -1), dpipd_template.contiguous().view(nb, nele, nazi, -1).permute(0, 3, 1, 2).view(nb, nmic * nf, -1)) / (nmic * nf / 2)
            pred_spatial_spectrum = pred_spatial_spectrum.view(nb, nt, nele, nazi)
            pred_DOAs = torch.zeros((nb, nt, 2, self.max_num_sources), dtype=torch.float32, requires_grad=False)
            pred_VADs = torch.zeros((nb, nt, self.max_num_sources), dtype=torch.float32, requires_grad=False)
            pred_DOAs = pred_DOAs
            pred_VADs = pred_VADs
            for source_idx in range(self.max_num_sources):
                map = torch.bmm(pred_rebatch.contiguous().view(nb, nt, -1), dpipd_template.contiguous().view(nb, nele, nazi, -1).permute(0, 3, 1, 2).view(nb, nmic * nf, -1)) / (nmic * nf / 2)
                map = map.view(nb, nt, nele, nazi)
                max_flat_idx = map.reshape((nb, nt, -1)).argmax(2)
                ele_max_idx, azi_max_idx = np.unravel_index(max_flat_idx.cpu().numpy(), map.shape[2:])
                ele_candidate = np.linspace(np.pi / 2, np.pi / 2, nele)
                azi_candidate = np.linspace(0, np.pi, nazi)
                pred_DOA = np.stack((ele_candidate[ele_max_idx], azi_candidate[azi_max_idx]), axis=-1)
                pred_DOA = torch.from_numpy(pred_DOA)
                pred_DOA = pred_DOA
                pred_DOAs[:, :, :, source_idx] = pred_DOA
                max_dpipd_template = torch.zeros((nb, nt, nf, nmic), dtype=torch.float32, requires_grad=False)
                max_dpipd_template = max_dpipd_template
                for b_idx in range(nb):
                    for t_idx in range(nt):
                        max_dpipd_template[b_idx, t_idx, :, :] = dpipd_template[b_idx, ele_max_idx[b_idx, t_idx], azi_max_idx[b_idx, t_idx], :, :] * 1.0
                        ratio = torch.sum(max_dpipd_template[b_idx, t_idx, :, :] * pred_rebatch[b_idx, t_idx, :, :]) / torch.sum(max_dpipd_template[b_idx, t_idx, :, :] * max_dpipd_template[b_idx, t_idx, :, :])
                        max_dpipd_template[b_idx, t_idx, :, :] = ratio * max_dpipd_template[b_idx, t_idx, :, :]
                        if self.source_num_mode == 'KNum':
                            pred_VADs[b_idx, t_idx, source_idx] = 1
                        elif self.source_num_mode == 'UnkNum':
                            pred_VADs[b_idx, t_idx, source_idx] = ratio * 1
                pred_rebatch = pred_rebatch - max_dpipd_template
            pred_batch = [pred_DOAs]
            pred_batch += [pred_VADs]
            pred_batch += [pred_spatial_spectrum]
        if gt_batch is not None:
            if type(gt_batch) is list:
                for idx in range(len(gt_batch)):
                    gt_batch[idx] = gt_batch[idx].detach()
            else:
                gt_batch = gt_batch.detach()
        return pred_batch, gt_batch

    def evaluate(self, pred_batch=None, gt_batch=None, vad_TH=[0.001, 0.5], idx=None):
        """
		evaluate the performance of DOA estimation
	    """
        ae_mode = ['azi']
        doa_gt = gt_batch[0] * 180 / np.pi
        doa_est = pred_batch[0] * 180 / np.pi
        vad_gt = gt_batch[-1]
        vad_est = pred_batch[-2]
        metric = {}
        if idx != None:
            np.save('./results/' + str(idx) + '_doagt', doa_gt.cpu().numpy())
            np.save('./results/' + str(idx) + '_doaest', doa_est.cpu().numpy())
            np.save('./results/' + str(idx) + '_vadgt', vad_gt.cpu().numpy())
            np.save('./results/' + str(idx) + '_vadest', vad_est.cpu().numpy())
            np.save('./results/' + str(idx) + '_ipd', pred_batch[-1].cpu().numpy())
        metric['ACC'], metric['MDR'], metric['FAR'], metric['MAE'], metric['RMSE'] = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=ae_mode, ae_TH=10, useVAD=True, vad_TH=vad_TH)
        return metric


class FakeModule(torch.nn.Module):

    def __init__(self, module: 'LightningModule') ->None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module.predict_step(x, 0)


def complex_conjugate_multiplication(x, y):
    return torch.stack([x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1], x[..., 1] * y[..., 0] - x[..., 0] * y[..., 1]], dim=-1)


class GCC(nn.Module):
    """ Compute the Generalized Cross Correlation of the inputs.
	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K).
	You can use tau_max to output only the central part of the GCCs and transform='PHAT' to use the PHAT transform.
	"""

    def __init__(self, N, K, tau_max=None, transform=None):
        assert transform is None or transform == 'PHAT', "Only the 'PHAT' transform is implemented"
        assert tau_max is None or tau_max <= K // 2
        super(GCC, self).__init__()
        self.K = K
        self.N = N
        self.tau_max = tau_max if tau_max is not None else K // 2
        self.transform = transform

    def forward(self, x):
        x_fft_c = torch.fft.rfft(x)
        x_fft = torch.stack((x_fft_c.real, x_fft_c.imag), -1)
        if self.transform == 'PHAT':
            mod = torch.sqrt(complex_conjugate_multiplication(x_fft, x_fft))[..., 0]
            mod += 1e-12
            x_fft /= mod.reshape(tuple(x_fft.shape[:-1]) + (1,))
        gcc = torch.empty(list(x_fft.shape[0:-3]) + [self.N, self.N, 2 * self.tau_max + 1], device=x.device)
        for n in range(self.N):
            gcc_fft_batch = complex_conjugate_multiplication(x_fft[..., n, :, :].unsqueeze(-3), x_fft)
            gcc_fft_batch_c = torch.complex(gcc_fft_batch[..., 0], gcc_fft_batch[..., 1])
            gcc_batch = torch.fft.irfft(gcc_fft_batch_c)
            gcc[..., n, :, 0:self.tau_max + 1] = gcc_batch[..., 0:self.tau_max + 1]
            gcc[..., n, :, -self.tau_max:] = gcc_batch[..., -self.tau_max:]
        return gcc


class SRP_map(nn.Module):
    """ Compute the SRP-PHAT maps from the GCCs taken as input.
	In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K), the
	desired resolution of the maps (resTheta and resPhi), the microphone positions relative to the center of the
	array (rn) and the sampling frequency (fs).
	With normalize=True (default) each map is normalized to ethe range [-1,1] approximately
	"""

    def __init__(self, N, K, resTheta, resPhi, rn, fs, c=343.0, normalize=True, thetaMax=np.pi / 2):
        super(SRP_map, self).__init__()
        self.N = N
        self.K = K
        self.resTheta = resTheta
        self.resPhi = resPhi
        self.fs = float(fs)
        self.normalize = normalize
        self.cross_idx = np.stack([np.kron(np.arange(N, dtype='int16'), np.ones(N, dtype='int16')), np.kron(np.ones(N, dtype='int16'), np.arange(N, dtype='int16'))])
        self.theta = np.linspace(0, thetaMax, resTheta)
        self.phi = np.linspace(-np.pi, np.pi, resPhi + 1)
        self.phi = self.phi[0:-1]
        self.IMTDF = np.empty((resTheta, resPhi, self.N, self.N))
        for k in range(self.N):
            for l in range(self.N):
                r = np.stack([np.outer(np.sin(self.theta), np.cos(self.phi)), np.outer(np.sin(self.theta), np.sin(self.phi)), np.tile(np.cos(self.theta), [resPhi, 1]).transpose()], axis=2)
                self.IMTDF[:, :, k, l] = np.dot(r, rn[l, :] - rn[k, :]) / c
        tau = np.concatenate([range(0, K // 2 + 1), range(-K // 2 + 1, 0)]) / float(fs)
        self.tau0 = np.zeros_like(self.IMTDF, dtype=np.int)
        for k in range(self.N):
            for l in range(self.N):
                for i in range(resTheta):
                    for j in range(resPhi):
                        self.tau0[i, j, k, l] = int(np.argmin(np.abs(self.IMTDF[i, j, k, l] - tau)))
        self.tau0[self.tau0 > K // 2] -= K
        self.tau0 = self.tau0.transpose([2, 3, 0, 1])

    def forward(self, x):
        tau0 = self.tau0
        tau0[tau0 < 0] += x.shape[-1]
        maps = torch.zeros(list(x.shape[0:-3]) + [self.resTheta, self.resPhi], device=x.device).float()
        for n in range(self.N):
            for m in range(self.N):
                maps += x[..., n, m, tau0[n, m, :, :]]
        if self.normalize:
            maps -= torch.mean(torch.mean(maps, -1, keepdim=True), -2, keepdim=True)
            maps += 1e-12
            maps /= torch.max(torch.max(maps, -1, keepdim=True)[0], -2, keepdim=True)[0]
        return maps


class SphericPad(nn.Module):
    """ Replication padding for time axis, reflect padding for the elevation and circular padding for the azimuth.
	The time padding is optional, do not use it with CausConv3d.
	"""

    def __init__(self, pad):
        super(SphericPad, self).__init__()
        if len(pad) == 4:
            self.padLeft, self.padRight, self.padTop, self.padBottom = pad
            self.padFront, self.padBack = 0, 0
        elif len(pad) == 6:
            self.padLeft, self.padRight, self.padTop, self.padBottom, self.padFront, self.padBack = pad
        else:
            raise Exception('Expect 4 or 6 values for padding (padLeft, padRight, padTop, padBottom, [padFront, padBack])')

    def forward(self, x):
        assert x.shape[-1] >= self.padRight and x.shape[-1] >= self.padLeft, 'Padding size should be less than the corresponding input dimension for the azimuth axis'
        if self.padBack > 0 or self.padFront > 0:
            x = F.pad(x, (0, 0, 0, 0, self.padFront, self.padBack), 'replicate')
        input_shape = x.shape
        x = x.view((x.shape[0], -1, x.shape[-2], x.shape[-1]))
        x = F.pad(x, (0, 0, self.padTop, self.padBottom), 'reflect')
        x = torch.cat((x[..., -self.padLeft:], x, x[..., :self.padRight]), dim=-1)
        return x.view((x.shape[0],) + input_shape[1:-2] + (x.shape[-2], x.shape[-1]))


class CausConv3d(nn.Module):
    """ Causal 3D Convolution for SRP-PHAT maps sequences
	"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausConv3d, self).__init__()
        self.pad = kernel_size[0] - 1
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=(self.pad, 0, 0))

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad, :, :]


class CausConv2d(nn.Module):
    """ Causal 2D Convolution for spectrograms and GCCs sequences
	"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausConv2d, self).__init__()
        self.pad = kernel_size[0] - 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.pad, 0))

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad, :]


class CausConv1d(nn.Module):
    """ Causal 1D Convolution
	"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad]


class CausCnnBlock1x1(nn.Module):

    def __init__(self, inplanes, planes, kernel=(1, 1), stride=(1, 1), padding=(0, 0)):
        super(CausCnnBlock1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out


class CausCnnBlock(nn.Module):
    """ 
    Function: Basic causal convolutional block
    """

    def __init__(self, inp_dim, out_dim, cnn_hidden_dim=128, kernel=(3, 3), stride=(1, 1), padding=(1, 2)):
        super(CausCnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp_dim, cnn_hidden_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(cnn_hidden_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 3))
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.pad = padding
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = out[:, :, :, :-self.pad[1]]
        out = self.pooling1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out[:, :, :, :-self.pad[1]]
        out = self.pooling2(out)
        out = self.conv3(out)
        out = out[:, :, :, :-self.pad[1]]
        out = self.tanh(out)
        return out


def pad_segments(x, seg_len):
    """ Pad the input tensor x to ensure the t-dimension is divisible by seg_len """
    nb, nt, nf, nc = x.shape
    pad_len = (seg_len - nt % seg_len) % seg_len
    if pad_len > 0:
        pad = torch.zeros(nb, pad_len, nf, nc, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    return x


def split_segments(x, seg_len):
    """ Split the input tensor x along the t-dimension into segments of length seg_len """
    nb, nt, nf, nc = x.shape
    x = pad_segments(x, seg_len)
    nt_padded = x.shape[1]
    x = x.reshape(nb, nt_padded // seg_len, seg_len, nf, nc)
    return x


class IPDnet(nn.Module):
    """
    The implementation of the IPDnet
    """

    def __init__(self, input_size=4, hidden_size=128, max_track=2, is_online=True, n_seg=312):
        super(IPDnet, self).__init__()
        self.is_online = is_online
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.block_1 = FNblock(input_size=self.input_size, hidden_size=self.hidden_size, add_skip_dim=self.input_size, is_online=self.is_online, is_first=True)
        self.block_2 = FNblock(input_size=self.hidden_size, hidden_size=self.hidden_size, add_skip_dim=self.input_size, is_online=self.is_online, is_first=False)
        self.cnn_out_dim = 2 * (input_size // 2 - 1) * max_track
        self.cnn_inp_dim = hidden_size + input_size
        self.conv = CausCnnBlock(inp_dim=self.cnn_inp_dim, out_dim=self.cnn_out_dim)
        self.n = n_seg

    def forward(self, x, offline_inference=False):
        x = x.permute(0, 3, 2, 1)
        nb, nt, nf, nc = x.shape
        ou_frame = nt // 12
        if not self.is_online and offline_inference:
            x = split_segments(x, self.n)
            nb, nseg, seg_nt, nf, nc = x.shape
            x = x.reshape(nb * nseg, seg_nt, nf, nc)
            nb, nt, nf, nc = x.shape
        fb_skip = x.reshape(nb * nt, nf, nc)
        nb_skip = x.permute(0, 2, 1, 3).reshape(nb * nf, nt, nc)
        x = self.block_1(x, fb_skip=fb_skip, nb_skip=nb_skip)
        x = self.block_2(x, fb_skip=fb_skip, nb_skip=nb_skip)
        nb, nt, nf, nc = x.shape
        x = x.permute(0, 3, 2, 1)
        nt2 = nt // 12
        x = self.conv(x).permute(0, 3, 2, 1).reshape(nb, nt2, nf, 2, -1).permute(0, 1, 3, 2, 4)
        if not self.is_online and offline_inference:
            x = x.reshape(nb // nseg, nt2 * nseg, 2, nf * 2, -1).permute(0, 1, 3, 4, 2)
            output = x[:, :ou_frame, :, :, :]
        else:
            output = x.reshape(nb, nt2, 2, nf * 2, -1).permute(0, 1, 3, 4, 2)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CausCnnBlock,
     lambda: ([], {'inp_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (CausCnnBlock1x1,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CausConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CausConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GCC,
     lambda: ([], {'N': 4, 'K': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IPDnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (STFT,
     lambda: ([], {'win_len': 4, 'win_shift_ratio': 4, 'nfft': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SourceDetectLocalize,
     lambda: ([], {'max_num_sources': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_Audio_WestlakeU_FN_SSL(_paritybench_base):
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

