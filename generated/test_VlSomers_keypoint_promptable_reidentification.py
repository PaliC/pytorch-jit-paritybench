import sys
_module = sys.modules[__name__]
del sys
demo = _module
main = _module
setup = _module
torchreid = _module
data = _module
data_augmentation = _module
batch_wise_inter_person_occlusion = _module
resize = _module
datamanager = _module
datasets = _module
dataset = _module
image = _module
cuhk01 = _module
cuhk02 = _module
cuhk03 = _module
dukemtmcreid = _module
grid = _module
ilids = _module
market1501 = _module
motchallenge = _module
msmt17 = _module
occluded_dukemtmc = _module
occluded_posetrack21 = _module
occluded_reid = _module
p_ETHZ = _module
p_dukemtmc_reid = _module
partial_ilids = _module
partial_reid = _module
prid = _module
viper = _module
keypoints_to_masks = _module
video = _module
dukemtmcvidreid = _module
ilidsvid = _module
mars = _module
prid2011 = _module
masks_transforms = _module
coco_keypoints_transforms = _module
keypoints_transform = _module
mask_transform = _module
pcb_transforms = _module
pifpaf_mask_transform = _module
sampler = _module
transforms = _module
engine = _module
engine = _module
part_based_engine = _module
softmax = _module
triplet = _module
softmax = _module
triplet = _module
hyperparameter = _module
GiLt_loss = _module
losses = _module
body_part_attention_loss = _module
cross_entropy_loss = _module
hard_mine_triplet_loss = _module
inter_parts_triplet_loss = _module
part_averaged_triplet_loss = _module
part_individual_triplet_loss = _module
part_max_min_triplet_loss = _module
part_max_triplet_loss = _module
part_min_triplet_loss = _module
part_random_max_min_triplet_loss = _module
metrics = _module
accuracy = _module
distance = _module
rank = _module
rank_cylib = _module
test_cython = _module
models = _module
compact_bilinear_pooling = _module
densenet = _module
hacnn = _module
hrnet = _module
inceptionresnetv2 = _module
inceptionv4 = _module
kpr = _module
mlfn = _module
mobilenetv2 = _module
mudeep = _module
nasnet = _module
osnet = _module
osnet_ain = _module
pcb = _module
promptable_solider = _module
promptable_timm_swin = _module
promptable_timm_vit = _module
promptable_transformer_backbone = _module
promptable_vit = _module
pvpm = _module
resnet = _module
resnet_fastreid = _module
resnet_ibn_a = _module
resnet_ibn_b = _module
resnetmid = _module
sam = _module
senet = _module
shufflenet = _module
shufflenetv2 = _module
solider = _module
backbones = _module
resnet = _module
resnet_ibn_a = _module
swin_transformer = _module
transformer_layers = _module
vit_pytorch = _module
configs = _module
defaults = _module
make_model = _module
squeezenet = _module
transreid = _module
resnet = _module
vit_pytorch = _module
make_model = _module
xception = _module
optim = _module
lr_scheduler = _module
optimizer = _module
radam = _module
schedulers = _module
cosine_lr = _module
scheduler = _module
scripts = _module
builder = _module
default_config = _module
tools = _module
compute_mean_std = _module
dataset_converters = _module
extract_part_based_features = _module
feature_extractor = _module
utils = _module
avgmeter = _module
constants = _module
distribution = _module
engine_state = _module
imagetools = _module
logging = _module
deprecated_loggers = _module
logger = _module
model_complexity = _module
reidtools = _module
rerank = _module
tensortools = _module
tools = _module
torch_receptive_field = _module
receptive_field = _module
torchtools = _module
visualization = _module
display_batch_triplets = _module
display_kpr_samples = _module
embeddings_projection = _module
feature_map_visualization = _module
visualize_query_gallery_rankings = _module
writer = _module

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


import math


import random


import torch.nn.functional as F


from scipy.ndimage import shift


import copy


from copy import deepcopy


from typing import Any


import re


import pandas as pd


from torch.utils.data import Dataset


from math import ceil


from torch.utils.data.dataloader import default_collate


from torch.utils.data.dataloader import DataLoader


from abc import abstractmethod


import logging


from abc import ABC


from abc import ABCMeta


from collections import OrderedDict


from torch import nn


from collections import defaultdict


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from torch.nn import functional as F


from torch.cuda import amp


import torch.nn as nn


from torch.nn import CrossEntropyLoss


import warnings


import types


from torch.autograd import Function


from torch.utils import model_zoo


import torch._utils


import torch.utils.model_zoo as model_zoo


from torchvision.ops import FeaturePyramidNetwork


from torch import nn as nn


from functools import partial


from itertools import repeat


import collections.abc


import torchvision


from typing import Sequence


import torch.utils.checkpoint as cp


from torch.nn import Module as BaseModule


from torch.nn import ModuleList


from torch.nn import Sequential


from torch.nn import Linear


from torch import Tensor


import collections.abc as container_abcs


from torch.optim.optimizer import Optimizer


from typing import Dict


import torchvision.transforms as T


import time


import matplotlib.pyplot as plt


from typing import Optional


from pandas.io.json._normalize import nested_to_record


from torch.utils.tensorboard import SummaryWriter


from collections import namedtuple


from functools import lru_cache


from torch.autograd import Variable


import itertools


import collections


from sklearn.decomposition import PCA


def cv2_load_image(file_path):
    file_path = str(file_path)
    image = cv2.imread(str(file_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class EngineDatapipe(Dataset):

    def __init__(self, model) ->None:
        self.model = model
        self.image_filepaths = None
        self.img_metadatas = None
        self.detections = None

    def update(self, image_filepaths: 'dict', img_metadatas, detections):
        del self.img_metadatas
        del self.detections
        self.image_filepaths = image_filepaths
        self.img_metadatas = img_metadatas
        self.detections = detections

    def __len__(self):
        if self.model.level == 'detection':
            return len(self.detections)
        elif self.model.level == 'image':
            return len(self.img_metadatas)
        else:
            raise ValueError(f"You should provide the appropriate level for you module not '{self.model.level}'")

    def __getitem__(self, idx):
        if self.model.level == 'detection':
            detection = self.detections.iloc[idx]
            metadata = self.img_metadatas.loc[detection.image_id]
            image = cv2_load_image(self.image_filepaths[metadata.name])
            sample = detection.name, self.model.preprocess(image=image, detection=detection, metadata=metadata)
            return sample
        elif self.model.level == 'image':
            metadata = self.img_metadatas.iloc[idx]
            if self.detections is not None and len(self.detections) > 0:
                detections = self.detections[self.detections.image_id == metadata.name]
            else:
                detections = self.detections
            image = cv2_load_image(self.image_filepaths[metadata.name])
            sample = self.img_metadatas.index[idx], self.model.preprocess(image=image, detections=detections, metadata=metadata)
            return sample
        else:
            raise ValueError('Please provide appropriate level.')


class ImageLevelModule(Module):
    """Abstract class to implement a module that operates directly on images.

    This can for example be a bounding box detector, or a bottom-up
    pose estimator (which outputs keypoints directly).

    The functions to implement are
     - __init__, which can take any configuration needed
     - preprocess
     - process
     - datapipe (optional) : returns an object which will be used to create the pipeline.
                            (Only modify this if you know what you're doing)
     - dataloader (optional) : returns a dataloader for the datapipe

     You should also provide the following class properties :
      - input_columns : what info you need for the detections
      - output_columns : what info you will provide when called
      - collate_fn (optional) : the function that will be used for collating the inputs
                                in a batch. (Default : pytorch collate function)

     A description of the expected behavior is provided below.
    """
    collate_fn = default_collate
    input_columns = None
    output_columns = None

    @abstractmethod
    def __init__(self, batch_size: 'int'):
        """Init function

        The arguments to this function are completely free
        and will be provided by a configuration file.

        You should call the __init__ function from the super() class.
        """
        self.batch_size = batch_size
        self._datapipe = None

    @abstractmethod
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series') ->Any:
        """Adapts the default input to your specific case.

        Args:
            image: a numpy array of the current image
            detections: a DataFrame containing all the detections pertaining to a single
                        image
            metadata: additional information about the image

        Returns:
            preprocessed_sample: input for the process function
        """
        pass

    @abstractmethod
    def process(self, batch: 'Any', detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        """The main processing function. Runs on GPU.

        Args:
            batch: The batched outputs of `preprocess`
            detections: The previous detections.
            metadatas: The previous image metadatas

        Returns:
            output : Either a DataFrame containing the new/updated detections
                    or a tuple containing detections and metadatas (in that order)
                    The DataFrames can be either a list of Series, a list of DataFrames
                    or a single DataFrame. The returned objects will be aggregated
                    automatically according to the `name` of the Series/`index` of
                    the DataFrame. **It is thus mandatory here to name correctly
                    your series or index your dataframes.**
                    The output will override the previous detections
                    with the same name/index.
        """
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "'TrackingEngine'"):
        datapipe = self.datapipe
        return DataLoader(dataset=datapipe, batch_size=self.batch_size, collate_fn=type(self).collate_fn, num_workers=engine.num_workers, persistent_workers=False)


CONCAT_PARTS = 'conct'


FOREGROUND = 'foreg'


GLOBAL = 'globl'


PARTS = 'parts'


def replace_values(input, mask, value):
    output = input * ~mask + mask * value
    return output


def masked_mean(input, mask):
    """ output -1 where mean couldn't be computed """
    valid_input = input * mask
    mean_weights = mask.sum(0)
    mean_weights = mean_weights + (mean_weights == 0)
    pairwise_dist = valid_input.sum(0) / mean_weights
    invalid_pairs = mask.sum(dim=0) == 0
    valid_pairwise_dist = replace_values(pairwise_dist, invalid_pairs, -1)
    return valid_pairwise_dist


class PartAveragedTripletLoss(nn.Module):
    """Compute the part-averaged triplet loss as described in our paper:
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    This class provides a generic implementation of the batch-hard triplet loss for part-based models, i.e. models
    outputting multiple embeddings (part-based/local representations) per input sample/image.
    When K=1 parts are provided and the parts_visiblity scores are set to one (or not provided), this implementation is
    strictly equal to the standard batch-hard triplet loss described in:
    'Alexander Hermans, Lucas Beyer, and Bastian Leibe. In Defense of the Triplet Loss for Person Re-Identification.'
    It is therefore valid to use this implementation for global embeddings too.
    Part-based distances are combined into a global sample-to-sample distance using a 'mean' operation.
    Other subclasses of PartAveragedTripletLoss provide different strategies to combine local distances into a global
    one.
    This implementation is optimized, using only tensors operations and no Python loops.
    """

    def __init__(self, margin=0.3, epsilon=1e-16, writer=None):
        super(PartAveragedTripletLoss, self).__init__()
        self.margin = margin
        self.writer = writer
        self.batch_debug = False
        self.imgs = None
        self.masks = None
        self.epsilon = epsilon

    def forward(self, part_based_embeddings, labels, parts_visibility=None):
        """
        The part averaged triplet loss is computed in three steps.
        Firstly, we compute the part-based pairwise distance matrix of size [K, N, N] for the K parts and the N 
        training samples.
        Secondly we compute the (samples) pairwise distance matrix of size [N, N] by combining the part-based distances.
        The part-based distances can be combined by averaging, max, min, etc.
        Thirdly, we compute the standard batch-hard triplet loss using the pairwise distance matrix.
        Compared to a standard triplet loss implementation, some entries in the pairwise distance matrix can have a
        value of -1. These entries correspond to pairs of samples that could not be compared, because there was no
        common visible parts for instance. Such pairs should be ignored for computing the batch hard triplets.
        
        Args:
            part_based_embeddings (torch.Tensor): feature matrix with shape (batch_size, parts_num, feat_dim).
            labels (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        part_based_pairwise_dist = self._part_based_pairwise_distance_matrix(part_based_embeddings.transpose(1, 0), squared=False)
        if parts_visibility is not None:
            parts_visibility = parts_visibility.t()
            valid_part_based_pairwise_dist_mask = parts_visibility.unsqueeze(1) * parts_visibility.unsqueeze(2)
            if valid_part_based_pairwise_dist_mask.dtype is not torch.bool:
                valid_part_based_pairwise_dist_mask = torch.sqrt(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist_mask = None
        pairwise_dist = self._combine_part_based_dist_matrices(part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels)
        return self._hard_mine_triplet_loss(pairwise_dist, labels, self.margin)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is not None:
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
            pairwise_dist = masked_mean(part_based_pairwise_dist, valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist = part_based_pairwise_dist
            pairwise_dist = valid_part_based_pairwise_dist.mean(0)
        return pairwise_dist

    def _part_based_pairwise_distance_matrix(self, embeddings, squared=False):
        """
        embeddings.shape = (K, N, C)
        ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
        """
        dot_product = torch.matmul(embeddings, embeddings.transpose(2, 1))
        square_sum = dot_product.diagonal(dim1=1, dim2=2)
        distances = square_sum.unsqueeze(2) - 2 * dot_product + square_sum.unsqueeze(1)
        distances = F.relu(distances)
        if not squared:
            mask = torch.eq(distances, 0).float()
            distances = distances + mask * self.epsilon
            distances = torch.sqrt(distances)
            distances = distances * (1 - mask)
        return distances

    def _hard_mine_triplet_loss(self, batch_pairwise_dist, labels, margin):
        """
        A generic implementation of the batch-hard triplet loss.
        K (part-based) distance matrix between N samples are provided in tensor 'batch_pairwise_dist' of size [K, N, N].
        The standard batch-hard triplet loss is then computed for each of the K distance matrix, yielding a total of KxN
        triplet losses.
        When a pairwise distance matrix of size [1, N, N] is provided with K=1, this function behave like a standard
        batch-hard triplet loss.
        When a pairwise distance matrix of size [K, N, N] is provided, this function will apply the batch-hard triplet
        loss strategy K times, i.e. one time for each of the K part-based distance matrix. It will then average all
        KxN triplet losses for all K parts into one loss value.
        For the part-averaged triplet loss described in the paper, all part-based distance are first averaged before
        calling this function, and a pairwise distance matrix of size [1, N, N] is provided here.
        When the triplet loss is applied individually for each part, without considering the global/combined distance
        between two training samples (as implemented by 'PartIndividualTripletLoss'), then a (part-based) pairwise
        distance matrix of size [K, N, N] is given as input.
        Compute distance matrix; i.e. for each anchor a_i with i=range(0, batch_size) :
        - find the (a_i,p_i) pair with greatest distance s.t. a_i and p_i have the same label
        - find the (a_i,n_i) pair with smallest distance s.t. a_i and n_i have different label
        - compute triplet loss for each triplet (a_i, p_i, n_i), average them
        Source :
        - https://github.com/lyakaap/NetVLAD-pytorch/blob/master/hard_triplet_loss.py
        - https://github.com/Yuol96/pytorch-triplet-loss/blob/master/model/triplet_loss.py
        Args:
            batch_pairwise_dist: pairwise distances between samples, of size (K, N, N). A value of -1 means no distance
                could be computed between the two sample, that pair should therefore not be considered for triplet
                mining.
            labels: id labels for the batch, of size (N,)
        Returns:
            triplet_loss: scalar tensor containing the batch hard triplet loss, which is the result of the average of a
                maximum of KxN triplet losses. Triplets are generated for anchors with at least one valid negative and
                one valid positive. Invalid negatives and invalid positives are marked with a -1 distance in
                batch_pairwise_dist input tensor.
            trivial_triplets_ratio: scalar between [0, 1] indicating the ratio of hard triplets that are 'trivial', i.e.
                for which the triplet loss value is 0 because the margin condition is already satisfied.
            valid_triplets_ratio: scalar between [0, 1] indicating the ratio of hard triplets that are valid. A triplet 
                is invalid if the anchor could not be compared with any positive or negative sample. Two samples cannot 
                be compared if they have no mutually visible parts (therefore no distance could be computed).
        """
        max_value = torch.finfo(batch_pairwise_dist.dtype).max
        valid_pairwise_dist_mask = batch_pairwise_dist != float(-1)
        self.writer.update_invalid_pairwise_distances_count(batch_pairwise_dist)
        mask_anchor_positive = self._get_anchor_positive_mask(labels).unsqueeze(0)
        mask_anchor_positive = mask_anchor_positive * valid_pairwise_dist_mask
        valid_positive_dist = batch_pairwise_dist * mask_anchor_positive.float() - (~mask_anchor_positive).float()
        hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=-1)
        mask_anchor_negative = self._get_anchor_negative_mask(labels).unsqueeze(0)
        mask_anchor_negative = mask_anchor_negative * valid_pairwise_dist_mask
        valid_negative_dist = batch_pairwise_dist * mask_anchor_negative.float() + (~mask_anchor_negative).float() * max_value
        hardest_negative_dist, _ = torch.min(valid_negative_dist, dim=-1)
        valid_hardest_positive_dist_mask = hardest_positive_dist != -1
        valid_hardest_negative_dist_mask = hardest_negative_dist != max_value
        valid_triplets_mask = valid_hardest_positive_dist_mask * valid_hardest_negative_dist_mask
        hardest_dist = torch.stack([hardest_positive_dist, hardest_negative_dist], 2)
        valid_hardest_dist = hardest_dist[valid_triplets_mask, :]
        if valid_hardest_dist.nelement() == 0:
            warnings.warn('CRITICAL WARNING: no valid triplets were generated for current batch')
            return None
        if self.margin > 0:
            triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.hard_margin_triplet_loss(margin, valid_hardest_dist, valid_triplets_mask)
        else:
            triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.soft_margin_triplet_loss(0.3, valid_hardest_dist, valid_triplets_mask)
        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def hard_margin_triplet_loss(self, margin, valid_hardest_dist, valid_triplets_mask):
        triplet_losses = F.relu(valid_hardest_dist[:, 0] - valid_hardest_dist[:, 1] + margin)
        triplet_loss = torch.mean(triplet_losses)
        trivial_triplets_ratio = (triplet_losses == 0.0).sum() / triplet_losses.nelement()
        valid_triplets_ratio = valid_triplets_mask.sum() / valid_triplets_mask.nelement()
        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def soft_margin_triplet_loss(self, margin, valid_hardest_dist, valid_triplets_mask):
        triplet_losses = F.relu(valid_hardest_dist[:, 0] - valid_hardest_dist[:, 1] + margin)
        hard_margin_triplet_loss = torch.mean(triplet_losses)
        trivial_triplets_ratio = (triplet_losses == 0.0).sum() / triplet_losses.nelement()
        valid_triplets_ratio = valid_triplets_mask.sum() / valid_triplets_mask.nelement()
        y = valid_hardest_dist[:, 0].new().resize_as_(valid_hardest_dist[:, 0]).fill_(1)
        soft_margin_triplet_loss = F.soft_margin_loss(valid_hardest_dist[:, 1] - valid_hardest_dist[:, 0], y)
        if soft_margin_triplet_loss == float('Inf'):
            None
            return hard_margin_triplet_loss, trivial_triplets_ratio, valid_triplets_ratio
        return soft_margin_triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    @staticmethod
    def _get_anchor_positive_mask(labels):
        """
        To be a valid positive pair (a,p) :
            - a and p are different embeddings
            - a and p have the same label
        """
        indices_equal_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.get_device() if labels.is_cuda else None)
        indices_not_equal_mask = ~indices_equal_mask
        labels_equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        mask_anchor_positive = indices_not_equal_mask * labels_equal_mask
        return mask_anchor_positive

    @staticmethod
    def _get_anchor_negative_mask(labels):
        """
        To be a valid negative pair (a,n) :
            - a and n have different labels (and therefore are different embeddings)
        """
        labels_not_equal_mask = torch.ne(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
        return labels_not_equal_mask


class InterPartsTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(InterPartsTripletLoss, self).__init__(**kwargs)

    def forward(self, body_parts_features, targets, n_iter=0, parts_visibility=None):
        body_parts_dist_matrices = self.compute_mixed_body_parts_dist_matrices(body_parts_features)
        return self.hard_mine_triplet_loss(body_parts_dist_matrices, targets)

    def compute_mixed_body_parts_dist_matrices(self, body_parts_features):
        body_parts_features = body_parts_features.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        body_parts_dist_matrices = self._part_based_pairwise_distance_matrix(body_parts_features, False, self.epsilon).squeeze()
        return body_parts_dist_matrices

    def hard_mine_triplet_loss(self, dist, targets):
        nm = dist.shape[0]
        n = targets.size(0)
        m = int(nm / n)
        expanded_targets = targets.repeat(m).expand(nm, -1)
        pids_mask = expanded_targets.eq(expanded_targets.t())
        body_parts_targets = []
        for i in range(0, m):
            body_parts_targets.append(torch.full_like(targets, i))
        body_parts_targets = torch.cat(body_parts_targets)
        expanded_body_parts_targets = body_parts_targets.expand(nm, -1)
        body_parts_mask = expanded_body_parts_targets.eq(expanded_body_parts_targets.t())
        mask_p = torch.logical_and(pids_mask, body_parts_mask)
        mask_n = pids_mask == 0
        dist_ap, dist_an = [], []
        for i in range(nm):
            i_pos_dist = dist[i][mask_p[i]]
            dist_ap.append(i_pos_dist.max().unsqueeze(0))
            i_neg_dist = dist[i][mask_n[i]]
            assert i_neg_dist.nelement() != 0, 'embedding %r should have at least one negative counterpart' % i
            dist_an.append(i_neg_dist.min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class PartIndividualTripletLoss(PartAveragedTripletLoss):
    """A triplet loss applied individually for each part, without considering the global/combined distance
        between two training samples. If the model outputs K embeddings (for K parts), this loss will compute the
        batch-hard triplet loss K times and output the average of them. With the part-averaged triplet loss, the global
        distance between two training samples is used in the triplet loss equation: that global distance is obtained by
        combining all K part-based distance between two samples into one value ('combining' = mean, max, min, etc).
        With the part-individual triplet loss, the triplet loss is applied only on local distance individually, i.e.,
        the distance between two local parts is used in the triplet loss equation. This part-individual triplet loss is
        therefore more sensitive to occluded parts (if 'valid_part_based_pairwise_dist_mask' is not used) and to
        non-discriminative parts, i.e. parts from two different identities having similar appearance.
        'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
        Source: https://github.com/VlSomers/bpbreid
        """

    def __init__(self, **kwargs):
        super(PartIndividualTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        """Do not combine part-based distance, simply return the input part-based pairwise distances, and optionally
        replace non-valid part-based distance with -1"""
        if valid_part_based_pairwise_dist_mask is not None:
            valid_part_based_pairwise_dist = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, -1)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist = part_based_pairwise_dist
        return valid_part_based_pairwise_dist


class PartMaxMinTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(PartMaxMinTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is not None:
            valid_part_based_pairwise_dist_for_max = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, -1)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist_for_max = part_based_pairwise_dist
        max_pairwise_dist, part_id_for_max = valid_part_based_pairwise_dist_for_max.max(0)
        if valid_part_based_pairwise_dist_mask is not None:
            max_value = torch.finfo(part_based_pairwise_dist.dtype).max
            valid_part_based_pairwise_dist_for_min = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, max_value)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist_for_min = part_based_pairwise_dist
        min_pairwise_dist, part_id_for_min = valid_part_based_pairwise_dist_for_min.min(0)
        labels_equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        pairwise_dist = max_pairwise_dist * labels_equal_mask + min_pairwise_dist * ~labels_equal_mask
        part_id = part_id_for_max * labels_equal_mask + part_id_for_min * ~labels_equal_mask
        if valid_part_based_pairwise_dist_mask is not None:
            invalid_pairwise_dist_mask = valid_part_based_pairwise_dist_mask.sum(dim=0) == 0
            pairwise_dist = replace_values(pairwise_dist, invalid_pairwise_dist_mask, -1)
        if part_based_pairwise_dist.shape[0] > 1:
            self.writer.used_parts_statistics(part_based_pairwise_dist.shape[0], part_id)
        return pairwise_dist


class PartMaxTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(PartMaxTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is not None:
            valid_part_based_pairwise_dist = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, -1)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist = part_based_pairwise_dist
        pairwise_dist, part_id = valid_part_based_pairwise_dist.max(0)
        parts_count = part_based_pairwise_dist.shape[0]
        if part_based_pairwise_dist.shape[0] > 1:
            self.writer.used_parts_statistics(parts_count, part_id)
        return pairwise_dist


class PartMinTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(PartMinTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is not None:
            max_value = torch.finfo(part_based_pairwise_dist.dtype).max
            valid_part_based_pairwise_dist = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, max_value)
            self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        else:
            valid_part_based_pairwise_dist = part_based_pairwise_dist
        pairwise_dist, part_id = valid_part_based_pairwise_dist.min(0)
        if valid_part_based_pairwise_dist_mask is not None:
            invalid_pairwise_dist_mask = valid_part_based_pairwise_dist_mask.sum(dim=0) == 0
            pairwise_dist = replace_values(pairwise_dist, invalid_pairwise_dist_mask, -1)
        parts_count = part_based_pairwise_dist.shape[0]
        if part_based_pairwise_dist.shape[0] > 1:
            self.writer.used_parts_statistics(parts_count, part_id)
        return pairwise_dist


class PartRandomMaxMinTripletLoss(PartAveragedTripletLoss):

    def __init__(self, **kwargs):
        super(PartRandomMaxMinTripletLoss, self).__init__(**kwargs)

    def _combine_part_based_dist_matrices(self, part_based_pairwise_dist, valid_part_based_pairwise_dist_mask, labels):
        if valid_part_based_pairwise_dist_mask is None:
            valid_part_based_pairwise_dist_mask = torch.ones(part_based_pairwise_dist.shape, dtype=torch.bool, device=labels.get_device() if labels.is_cuda else None)
        dropout_mask = torch.rand(size=valid_part_based_pairwise_dist_mask.shape, device=labels.get_device() if labels.is_cuda else None) > 0.5
        valid_part_based_pairwise_dist_mask *= dropout_mask
        valid_part_based_pairwise_dist_for_max = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, -1)
        self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        max_pairwise_dist, part_id_for_max = valid_part_based_pairwise_dist_for_max.max(0)
        max_value = torch.finfo(part_based_pairwise_dist.dtype).max
        valid_part_based_pairwise_dist_for_min = replace_values(part_based_pairwise_dist, ~valid_part_based_pairwise_dist_mask, max_value)
        self.writer.update_invalid_part_based_pairwise_distances_count(valid_part_based_pairwise_dist_mask)
        min_pairwise_dist, part_id_for_min = valid_part_based_pairwise_dist_for_min.min(0)
        labels_equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        pairwise_dist = max_pairwise_dist * labels_equal_mask + min_pairwise_dist * ~labels_equal_mask
        part_id = part_id_for_max * labels_equal_mask + part_id_for_min * ~labels_equal_mask
        invalid_pairwise_dist_mask = valid_part_based_pairwise_dist_mask.sum(dim=0) == 0
        pairwise_dist = replace_values(pairwise_dist, invalid_pairwise_dist_mask, -1)
        if part_based_pairwise_dist.shape[0] > 1:
            self.writer.used_parts_statistics(part_based_pairwise_dist.shape[0], part_id)
        return pairwise_dist


__body_parts_losses = {'part_averaged_triplet_loss': PartAveragedTripletLoss, 'part_max_triplet_loss': PartMaxTripletLoss, 'part_min_triplet_loss': PartMinTripletLoss, 'part_max_min_triplet_loss': PartMaxMinTripletLoss, 'part_random_max_min_triplet_loss': PartRandomMaxMinTripletLoss, 'inter_parts_triplet_loss': InterPartsTripletLoss, 'intra_parts_triplet_loss': PartIndividualTripletLoss}


def init_part_based_triplet_loss(name, **kwargs):
    """Initializes the part based triplet loss based on the part-based distance combination strategy."""
    avai_body_parts_losses = list(__body_parts_losses.keys())
    if name not in avai_body_parts_losses:
        raise ValueError('Invalid loss name. Received "{}", but expected to be one of {}'.format(name, avai_body_parts_losses))
    return __body_parts_losses[name](**kwargs)


class GiLtLoss(nn.Module):
    """ The Global-identity Local-triplet 'GiLt' loss as described in our paper:
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    The default weights for the GiLt strategy (as described in the paper) are provided in 'default_losses_weights': the
    identity loss is applied only on holistic embeddings and the triplet loss is applied only on part-based embeddings.
    'tr' denotes 'triplet' for the triplet loss and 'id' denotes 'identity' for the identity cross-entropy loss.
    """
    default_losses_weights = {GLOBAL: {'id': 1.0, 'tr': 0.0}, FOREGROUND: {'id': 1.0, 'tr': 0.0}, CONCAT_PARTS: {'id': 1.0, 'tr': 0.0}, PARTS: {'id': 0.0, 'tr': 1.0}}

    def __init__(self, losses_weights=None, use_visibility_scores=False, triplet_margin=0.3, loss_name='part_averaged_triplet_loss', use_gpu=False, num_classes=-1, writer=None):
        super().__init__()
        if losses_weights is None:
            losses_weights = self.default_losses_weights
        self.use_gpu = use_gpu
        self.pred_accuracy = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        if self.use_gpu:
            self.pred_accuracy = self.pred_accuracy
        self.losses_weights = losses_weights
        self.part_triplet_loss = init_part_based_triplet_loss(loss_name, margin=triplet_margin, writer=writer)
        self.identity_loss = CrossEntropyLoss(label_smooth=True)
        self.use_visibility_scores = use_visibility_scores

    def forward(self, embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids):
        """
        Keys in the input dictionaries are from {'globl', 'foreg', 'conct', 'parts'} and correspond to the different
        types of embeddings. In the documentation below, we denote the batch size by 'N' and the number of parts by 'K'.
        :param embeddings_dict: a dictionary of embeddings, where the keys are the embedding types and the values are
            Tensors of size [N, D] or [N, K*D] or [N, K, D].
        :param visibility_scores_dict: a dictionary of visibility scores, where the keys are the embedding types and the
            values are Tensors of size [N] or [N, K].
        :param id_cls_scores_dict: a dictionary of identity classification scores, where the keys are the embedding types
            and the values are Tensors of size [N, num_classes] or [N, K, num_classes]
        :param pids: A Tensor of size [N] containing the person IDs.
        :return: a tupel with the total combined loss and a dictionnary with performance information for each individual
            loss.
        """
        loss_summary = {}
        losses = []
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            loss_info = OrderedDict() if key not in loss_summary else loss_summary[key]
            ce_w = self.losses_weights[key]['id']
            if ce_w > 0:
                parts_id_loss, parts_id_accuracy = self.compute_id_cls_loss(id_cls_scores_dict[key], visibility_scores_dict[key], pids)
                losses.append((ce_w, parts_id_loss))
                loss_info['c'] = parts_id_loss
                loss_info['a'] = parts_id_accuracy
            loss_summary[key] = loss_info
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            loss_info = OrderedDict() if key not in loss_summary else loss_summary[key]
            tr_w = self.losses_weights[key]['tr']
            if tr_w > 0:
                parts_triplet_loss, parts_trivial_triplets_ratio, parts_valid_triplets_ratio = self.compute_triplet_loss(embeddings_dict[key], visibility_scores_dict[key], pids)
                losses.append((tr_w, parts_triplet_loss))
                loss_info['t'] = parts_triplet_loss
                loss_info['tt'] = parts_trivial_triplets_ratio
                loss_info['vt'] = parts_valid_triplets_ratio
            loss_summary[key] = loss_info
        if len(losses) == 0:
            return torch.tensor(0.0, device=pids.get_device() if pids.is_cuda else None), loss_summary
        else:
            loss = torch.stack([(weight * loss) for weight, loss in losses]).sum()
            return loss, loss_summary

    def compute_triplet_loss(self, embeddings, visibility_scores, pids):
        if self.use_visibility_scores:
            visibility = visibility_scores if len(visibility_scores.shape) == 2 else visibility_scores.unsqueeze(1)
        else:
            visibility = None
        embeddings = embeddings if len(embeddings.shape) == 3 else embeddings.unsqueeze(1)
        triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.part_triplet_loss(embeddings, pids, parts_visibility=visibility)
        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def compute_id_cls_loss(self, id_cls_scores, visibility_scores, pids):
        if len(id_cls_scores.shape) == 3:
            M = id_cls_scores.shape[1]
            id_cls_scores = id_cls_scores.flatten(0, 1)
            pids = pids.unsqueeze(1).expand(-1, M).flatten(0, 1)
            visibility_scores = visibility_scores.flatten(0, 1)
        weights = None
        if self.use_visibility_scores and visibility_scores.dtype is torch.bool:
            id_cls_scores = id_cls_scores[visibility_scores]
            pids = pids[visibility_scores]
        elif self.use_visibility_scores and visibility_scores.dtype is not torch.bool:
            weights = visibility_scores
        cls_loss = self.identity_loss(id_cls_scores, pids, weights)
        accuracy = self.pred_accuracy(id_cls_scores, pids)
        return cls_loss, accuracy


PIXELS = 'pixls'


class BodyPartAttentionLoss(nn.Module):
    """ A body part attention loss as described in our paper
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    """

    def __init__(self, loss_type='cl', label_smoothing=0.1, use_gpu=False, best_pred_ratio=100, num_classes=-1):
        super().__init__()
        self.pred_accuracy = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.best_pred_ratio = best_pred_ratio
        self.loss_type = loss_type
        self.pred_accuracy = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        if use_gpu:
            self.pred_accuracy = self.pred_accuracy
        if loss_type == 'cl':
            self.part_prediction_loss = CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
        elif loss_type == 'fl':
            self.part_prediction_loss = FocalLoss(to_onehot_y=True, gamma=1.0, reduction='mean')
        elif loss_type == 'dl':
            self.part_prediction_loss = DiceLoss(to_onehot_y=True, softmax=True, reduction='mean')
        else:
            raise ValueError('Loss {} for part prediction is not supported'.format(loss_type))

    def forward(self, pixels_cls_scores, targets):
        """ Compute loss for body part attention prediction.
            Args:
                pixels_cls_scores [N, K, H, W]
                targets [N, H, W]
            Returns:
        """
        loss_summary = {}
        loss_summary[PIXELS] = OrderedDict()
        pixels_cls_loss, pixels_cls_accuracy = self.compute_pixels_cls_loss(pixels_cls_scores, targets)
        loss_summary[PIXELS]['c'] = pixels_cls_loss
        loss_summary[PIXELS]['a'] = pixels_cls_accuracy
        return pixels_cls_loss, loss_summary

    def compute_pixels_cls_loss(self, pixels_cls_scores, targets):
        if pixels_cls_scores.is_cuda:
            targets = targets
        pixels_cls_score_targets = targets.flatten()
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(0, 2)
        losses = self.part_prediction_loss(pixels_cls_scores, pixels_cls_score_targets)
        if self.loss_type == 'cl':
            filtered_losses = losses[torch.topk(losses, int(len(losses) * self.best_pred_ratio), largest=False).indices]
            loss = torch.mean(filtered_losses)
        else:
            loss = losses
        accuracy = self.pred_accuracy(pixels_cls_scores, pixels_cls_score_targets)
        return loss, accuracy.item()


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by
    
    .. math::
        \\begin{equation}
        (1 - \\eps) \\times y + \\frac{\\eps}{K},
        \\end{equation}

    where :math:`K` denotes the number of classes and :math:`\\eps` is a weight. When
    :math:`\\eps = 0`, the loss function reduces to the normal cross entropy.
    
    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, eps=0.1, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.eps = eps if label_smooth else 0
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weights=None):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        assert inputs.shape[0] == targets.shape[0]
        num_classes = inputs.shape[1]
        log_probs = self.logsoftmax(inputs)
        zeros = torch.zeros(log_probs.size())
        targets = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if inputs.is_cuda:
            targets = targets
        targets = (1 - self.eps) * targets + self.eps / num_classes
        if weights is not None:
            result = (-targets * log_probs).sum(dim=1)
            result = result * nn.functional.normalize(weights, p=1, dim=0)
            result = result.sum()
        else:
            result = (-targets * log_probs).mean(0).sum()
        return result


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        dist = self.compute_dist_matrix(inputs)
        return self.compute_hard_mine_triplet_loss(dist, inputs, targets)

    def compute_hard_mine_triplet_loss(self, dist, inputs, targets):
        n = inputs.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

    def compute_dist_matrix(self, inputs):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)
        dist = dist.sqrt()
        return dist


def CountSketchFn_backward(h, s, x_size, grad_output):
    s_view = (1,) * (len(x_size) - 1) + (x_size[-1],)
    s = s.view(s_view)
    h = h.view(s_view).expand(x_size)
    grad_x = grad_output.gather(-1, h)
    grad_x = grad_x * s
    return grad_x


def CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add=False):
    x_size = tuple(x.size())
    s_view = (1,) * (len(x_size) - 1) + (x_size[-1],)
    out_size = x_size[:-1] + (output_size,)
    s = s.view(s_view)
    xs = x * s
    h = h.view(s_view).expand(x_size)
    if force_cpu_scatter_add:
        out = x.new(*out_size).zero_().cpu()
        return out.scatter_add_(-1, h.cpu(), xs.cpu())
    else:
        out = x.new(*out_size).zero_()
        return out.scatter_add_(-1, h, xs)


class CountSketchFn(Function):

    @staticmethod
    def forward(ctx, h, s, output_size, x, force_cpu_scatter_add=False):
        x_size = tuple(x.size())
        ctx.save_for_backward(h, s)
        ctx.x_size = tuple(x.size())
        return CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add)

    @staticmethod
    def backward(ctx, grad_output):
        h, s = ctx.saved_variables
        grad_x = CountSketchFn_backward(h, s, ctx.x_size, grad_output)
        return None, None, None, grad_x


class CountSketch(nn.Module):
    """Compute the count sketch over an input signal.

    .. math::

        out_j = \\sum_{i : j = h_i} s_i x_i

    Args:
        input_size (int): Number of channels in the input array
        output_size (int): Number of channels in the output sketch
        h (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s (array, optional): Optional array of size input_size of -1 and 1.

    .. note::

        If h and s are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input: (...,input_size)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input_size, output_size, h=None, s=None):
        super(CountSketch, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        if h is None:
            h = torch.LongTensor(input_size).random_(0, output_size)
        if s is None:
            s = 2 * torch.Tensor(input_size).random_(0, 2) - 1

        def identity(self):
            return self
        h.float = types.MethodType(identity, h)
        h.double = types.MethodType(identity, h)
        self.register_buffer('h', h)
        self.register_buffer('s', s)

    def forward(self, x):
        x_size = list(x.size())
        assert x_size[-1] == self.input_size
        return CountSketchFn.apply(self.h, self.s, self.output_size, x)


def ComplexMultiply_forward(X_re, X_im, Y_re, Y_im):
    Z_re = torch.addcmul(X_re * Y_re, -1, X_im, Y_im)
    Z_im = torch.addcmul(X_re * Y_im, 1, X_im, Y_re)
    return Z_re, Z_im


class CompactBilinearPoolingFn(Function):

    @staticmethod
    def forward(ctx, h1, s1, h2, s2, output_size, x, y, force_cpu_scatter_add=False):
        ctx.save_for_backward(h1, s1, h2, s2, x, y)
        ctx.x_size = tuple(x.size())
        ctx.y_size = tuple(y.size())
        ctx.force_cpu_scatter_add = force_cpu_scatter_add
        ctx.output_size = output_size
        px = CountSketchFn_forward(h1, s1, output_size, x, force_cpu_scatter_add)
        fx = torch.rfft(px, 1)
        re_fx = fx.select(-1, 0)
        im_fx = fx.select(-1, 1)
        del px
        py = CountSketchFn_forward(h2, s2, output_size, y, force_cpu_scatter_add)
        fy = torch.rfft(py, 1)
        re_fy = fy.select(-1, 0)
        im_fy = fy.select(-1, 1)
        del py
        re_prod, im_prod = ComplexMultiply_forward(re_fx, im_fx, re_fy, im_fy)
        re = torch.irfft(torch.stack((re_prod, im_prod), re_prod.dim()), 1, signal_sizes=(output_size,))
        return re

    @staticmethod
    def backward(ctx, grad_output):
        h1, s1, h2, s2, x, y = ctx.saved_tensors
        px = CountSketchFn_forward(h1, s1, ctx.output_size, x, ctx.force_cpu_scatter_add)
        py = CountSketchFn_forward(h2, s2, ctx.output_size, y, ctx.force_cpu_scatter_add)
        grad_output = grad_output.contiguous()
        grad_prod = torch.rfft(grad_output, 1)
        grad_re_prod = grad_prod.select(-1, 0)
        grad_im_prod = grad_prod.select(-1, 1)
        fy = torch.rfft(py, 1)
        re_fy = fy.select(-1, 0)
        im_fy = fy.select(-1, 1)
        del py
        grad_re_fx = torch.addcmul(grad_re_prod * re_fy, 1, grad_im_prod, im_fy)
        grad_im_fx = torch.addcmul(grad_im_prod * re_fy, -1, grad_re_prod, im_fy)
        grad_fx = torch.irfft(torch.stack((grad_re_fx, grad_im_fx), grad_re_fx.dim()), 1, signal_sizes=(ctx.output_size,))
        grad_x = CountSketchFn_backward(h1, s1, ctx.x_size, grad_fx)
        del re_fy, im_fy, grad_re_fx, grad_im_fx, grad_fx
        fx = torch.rfft(px, 1)
        re_fx = fx.select(-1, 0)
        im_fx = fx.select(-1, 1)
        del px
        grad_re_fy = torch.addcmul(grad_re_prod * re_fx, 1, grad_im_prod, im_fx)
        grad_im_fy = torch.addcmul(grad_im_prod * re_fx, -1, grad_re_prod, im_fx)
        grad_fy = torch.irfft(torch.stack((grad_re_fy, grad_im_fy), grad_re_fy.dim()), 1, signal_sizes=(ctx.output_size,))
        grad_y = CountSketchFn_backward(h2, s2, ctx.y_size, grad_fy)
        del re_fx, im_fx, grad_re_fy, grad_im_fy, grad_fy
        return None, None, None, None, None, grad_x, grad_y, None


class CompactBilinearPooling(nn.Module):
    """Compute the compact bilinear pooling between two input array x and y

    .. math::

        out = \\Psi (x,h_1,s_1) \\ast \\Psi (y,h_2,s_2)

    Args:
        input_size1 (int): Number of channels in the first input array
        input_size2 (int): Number of channels in the second input array
        output_size (int): Number of channels in the output array
        h1 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s1 (array, optional): Optional array of size input_size of -1 and 1.
        h2 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s2 (array, optional): Optional array of size input_size of -1 and 1.
        force_cpu_scatter_add (boolean, optional): Force the scatter_add operation to run on CPU for testing purposes

    .. note::

        If h1, s1, s2, h2 are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input 1: (...,input_size1)
        - Input 2: (...,input_size2)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input1_size, input2_size, output_size, h1=None, s1=None, h2=None, s2=None, force_cpu_scatter_add=False):
        super(CompactBilinearPooling, self).__init__()
        self.add_module('sketch1', CountSketch(input1_size, output_size, h1, s1))
        self.add_module('sketch2', CountSketch(input2_size, output_size, h2, s2))
        self.output_size = output_size
        self.force_cpu_scatter_add = force_cpu_scatter_add

    def forward(self, x, y=None):
        if y is None:
            y = x
        return CompactBilinearPoolingFn.apply(self.sketch1.h, self.sketch1.s, self.sketch2.h, self.sketch2.s, self.output_size, x, y, self.force_cpu_scatter_add)


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densely connected network.
    
    Reference:
        Huang et al. Densely Connected Convolutional Networks. CVPR 2017.

    Public keys:
        - ``densenet121``: DenseNet121.
        - ``densenet169``: DenseNet169.
        - ``densenet201``: DenseNet201.
        - ``densenet161``: DenseNet161.
        - ``densenet121_fc512``: DenseNet121 + FC.
    """

    def __init__(self, num_classes, loss, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, fc_dims=None, dropout_p=None, **kwargs):
        super(DenseNet, self).__init__()
        self.loss = loss
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = num_features
        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s, p):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        mid_channels = out_channels // 4
        self.stream1 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, p=1))
        self.stream2 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, p=1))
        self.stream3 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, p=1))
        self.stream4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), ConvBlock(in_channels, mid_channels, 1))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class InceptionB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        mid_channels = out_channels // 4
        self.stream1 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, s=2, p=1))
        self.stream2 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, p=1), ConvBlock(mid_channels, mid_channels, 3, s=2, p=1))
        self.stream3 = nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1), ConvBlock(in_channels, mid_channels * 2, 1))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y


class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        x = x.mean(1, keepdim=True)
        x = self.conv1(x)
        x = F.upsample(x, (x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float))

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta


class HACNN(nn.Module):
    """Harmonious Attention Convolutional Neural Network.

    Reference:
        Li et al. Harmonious Attention Network for Person Re-identification. CVPR 2018.

    Public keys:
        - ``hacnn``: HACNN.
    """

    def __init__(self, num_classes, loss='softmax', nchannels=[128, 256, 384], feat_dim=512, learn_region=True, use_gpu=True, **kwargs):
        super(HACNN, self).__init__()
        self.loss = loss
        self.learn_region = learn_region
        self.use_gpu = use_gpu
        self.conv = ConvBlock(3, 32, 3, s=2, p=1)
        self.inception1 = nn.Sequential(InceptionA(32, nchannels[0]), InceptionB(nchannels[0], nchannels[0]))
        self.ha1 = HarmAttn(nchannels[0])
        self.inception2 = nn.Sequential(InceptionA(nchannels[0], nchannels[1]), InceptionB(nchannels[1], nchannels[1]))
        self.ha2 = HarmAttn(nchannels[1])
        self.inception3 = nn.Sequential(InceptionA(nchannels[1], nchannels[2]), InceptionB(nchannels[2], nchannels[2]))
        self.ha3 = HarmAttn(nchannels[2])
        self.fc_global = nn.Sequential(nn.Linear(nchannels[2], feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU())
        self.classifier_global = nn.Linear(feat_dim, num_classes)
        if self.learn_region:
            self.init_scale_factors()
            self.local_conv1 = InceptionB(32, nchannels[0])
            self.local_conv2 = InceptionB(nchannels[0], nchannels[1])
            self.local_conv3 = InceptionB(nchannels[1], nchannels[2])
            self.fc_local = nn.Sequential(nn.Linear(nchannels[2] * 4, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU())
            self.classifier_local = nn.Linear(feat_dim, num_classes)
            self.feat_dim = feat_dim * 2
        else:
            self.feat_dim = feat_dim

    def init_scale_factors(self):
        self.scale_factors = []
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))

    def stn(self, x, theta):
        """Performs spatial transform
        
        x: (batch, channel, height, width)
        theta: (batch, 2, 3)
        """
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform_theta(self, theta_i, region_idx):
        """Transforms theta to include (s_w, s_h), resulting in (batch, 2, 3)"""
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:, :, :2] = scale_factors
        theta[:, :, -1] = theta_i
        if self.use_gpu:
            theta = theta
        return theta

    def forward(self, x):
        assert x.size(2) == 160 and x.size(3) == 64, 'Input size does not match, expected (160, 64) but got ({}, {})'.format(x.size(2), x.size(3))
        x = self.conv(x)
        x1 = self.inception1(x)
        x1_attn, x1_theta = self.ha1(x1)
        x1_out = x1 * x1_attn
        if self.learn_region:
            x1_local_list = []
            for region_idx in range(4):
                x1_theta_i = x1_theta[:, region_idx, :]
                x1_theta_i = self.transform_theta(x1_theta_i, region_idx)
                x1_trans_i = self.stn(x, x1_theta_i)
                x1_trans_i = F.upsample(x1_trans_i, (24, 28), mode='bilinear', align_corners=True)
                x1_local_i = self.local_conv1(x1_trans_i)
                x1_local_list.append(x1_local_i)
        x2 = self.inception2(x1_out)
        x2_attn, x2_theta = self.ha2(x2)
        x2_out = x2 * x2_attn
        if self.learn_region:
            x2_local_list = []
            for region_idx in range(4):
                x2_theta_i = x2_theta[:, region_idx, :]
                x2_theta_i = self.transform_theta(x2_theta_i, region_idx)
                x2_trans_i = self.stn(x1_out, x2_theta_i)
                x2_trans_i = F.upsample(x2_trans_i, (12, 14), mode='bilinear', align_corners=True)
                x2_local_i = x2_trans_i + x1_local_list[region_idx]
                x2_local_i = self.local_conv2(x2_local_i)
                x2_local_list.append(x2_local_i)
        x3 = self.inception3(x2_out)
        x3_attn, x3_theta = self.ha3(x3)
        x3_out = x3 * x3_attn
        if self.learn_region:
            x3_local_list = []
            for region_idx in range(4):
                x3_theta_i = x3_theta[:, region_idx, :]
                x3_theta_i = self.transform_theta(x3_theta_i, region_idx)
                x3_trans_i = self.stn(x2_out, x3_theta_i)
                x3_trans_i = F.upsample(x3_trans_i, (6, 7), mode='bilinear', align_corners=True)
                x3_local_i = x3_trans_i + x2_local_list[region_idx]
                x3_local_i = self.local_conv3(x3_local_i)
                x3_local_list.append(x3_local_i)
        x_global = F.avg_pool2d(x3_out, x3_out.size()[2:]).view(x3_out.size(0), x3_out.size(1))
        x_global = self.fc_global(x_global)
        if self.learn_region:
            x_local_list = []
            for region_idx in range(4):
                x_local_i = x3_local_list[region_idx]
                x_local_i = F.avg_pool2d(x_local_i, x_local_i.size()[2:]).view(x_local_i.size(0), -1)
                x_local_list.append(x_local_i)
            x_local = torch.cat(x_local_list, 1)
            x_local = self.fc_local(x_local)
        if not self.training:
            if self.learn_region:
                x_global = x_global / x_global.normalization(p=2, dim=1, keepdim=True)
                x_local = x_local / x_local.normalization(p=2, dim=1, keepdim=True)
                return torch.cat([x_global, x_local], 1)
            else:
                return x_global
        prelogits_global = self.classifier_global(x_global)
        if self.learn_region:
            prelogits_local = self.classifier_local(x_local)
        if self.loss == 'softmax':
            if self.learn_region:
                return prelogits_global, prelogits_local
            else:
                return prelogits_global
        elif self.loss == 'triplet':
            if self.learn_region:
                return (prelogits_global, prelogits_local), (x_global, x_local)
            else:
                return prelogits_global, x_global
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


BN_MOMENTUM = 0.1


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            None
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            None
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            None
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM), nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM), nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, enable_dim_reduction, dim_reduction_channels, img_size, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
        self.incre_modules, _, _ = self._make_head(pre_stage_channels)
        self.layers_out_channels = 1920
        self.dim_reduction_channels = dim_reduction_channels
        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=self.layers_out_channels, out_channels=self.dim_reduction_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(self.dim_reduction_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.enable_dim_reduction = enable_dim_reduction
        if self.enable_dim_reduction:
            self.feature_dim = self.dim_reduction_channels
        else:
            self.feature_dim = self.layers_out_channels
        self.spatial_feature_shape = int(img_size[0] / 4), int(img_size[1] / 4), self.feature_dim
        self.random_init()

    def _make_incre_channel_nin(self):
        head_channels = [128, 256, 512, 1024]
        incre_modules = []
        for i in range(3):
            incre_module = nn.Sequential(nn.Conv2d(in_channels=head_channels[i], out_channels=head_channels[i + 1], kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(head_channels[i + 1], momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
        return incre_modules

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], 1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)
        final_layer = nn.Sequential(nn.Conv2d(in_channels=head_channels[3] * head_block.expansion, out_channels=2048, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(2048, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        for i in range(len(self.incre_modules)):
            x[i] = self.incre_modules[i](x[i])
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x[0], x1, x2, x3], 1)
        if self.enable_dim_reduction:
            x = self.cls_head(x)
        return x

    def random_init(self):
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_param(self, pretrained_path):
        if not Path(pretrained_path).exists():
            raise FileNotFoundError(f'HRNet-W32-C pretrained weights not found under "{pretrained_path}", please download it first at https://github.com/HRNet/HRNet-Image-Classification or specify the correct weights dir location with the cfg.model.backbone_pretrained_path config.')
        pretrained_dict = torch.load(pretrained_path)
        None
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


pretrained_settings = {'xception': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth', 'input_space': 'RGB', 'input_size': [3, 299, 299], 'input_range': [0, 1], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'num_classes': 1000, 'scale': 0.8975}}}


class InceptionResNetV2(nn.Module):
    """Inception-ResNet-V2.

    Reference:
        Szegedy et al. Inception-v4, Inception-ResNet and the Impact of Residual
        Connections on Learning. AAAI 2017.

    Public keys:
        - ``inceptionresnetv2``: Inception-ResNet-V2.
    """

    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(InceptionResNetV2, self).__init__()
        self.loss = loss
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1536, num_classes)

    def load_imagenet_weights(self):
        settings = pretrained_settings['inceptionresnetv2']['imagenet']
        pretrain_dict = model_zoo.load_url(settings['url'])
        model_dict = self.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)

    def featuremaps(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):
    """Inception-v4.

    Reference:
        Szegedy et al. Inception-v4, Inception-ResNet and the Impact of Residual
        Connections on Learning. AAAI 2017.

    Public keys:
        - ``inceptionv4``: InceptionV4.
    """

    def __init__(self, num_classes, loss, **kwargs):
        super(InceptionV4, self).__init__()
        self.loss = loss
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(), Inception_A(), Inception_A(), Reduction_A(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Reduction_B(), Inception_C(), Inception_C(), Inception_C())
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        f = self.features(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class AfterPoolingDimReduceLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_p=None):
        super(AfterPoolingDimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, output_dim, bias=True))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.opout(p=dropout_p))
        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        if len(x.size()) == 3:
            N, K, _ = x.size()
            x = x.flatten(0, 1)
            x = self.layers(x)
            x = x.view(N, K, -1)
        else:
            x = self.layers(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


BACKGROUND = 'backg'


class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()
        self.in_dim = in_dim
        self.class_num = class_num
        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)
        self._init_params()

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


BN_BACKGROUND = 'bn_backg'


BN_CONCAT_PARTS = 'bn_conct'


BN_FOREGROUND = 'bn_foreg'


BN_GLOBAL = 'bn_globl'


BN_PARTS = 'bn_parts'


class BeforePoolingDimReduceLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(BeforePoolingDimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, 1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        return self.layers(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GlobalMaskWeightedPoolingHead(nn.Module):

    def __init__(self, depth, normalization='identity'):
        super().__init__()
        if normalization == 'identity':
            self.normalization = nn.Identity()
        elif normalization == 'batch_norm_3d':
            self.normalization = torch.nn.BatchNorm3d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_2d':
            self.normalization = torch.nn.BatchNorm2d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_1d':
            self.normalization = torch.nn.BatchNorm1d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            raise ValueError('normalization type {} not supported'.format(normalization))

    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)
        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = self.global_pooling(parts_features)
        parts_features = parts_features.view(N, M, -1)
        return parts_features

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GlobalAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveAvgPool2d((1, 1))


class MultiStageFusion(nn.Module):
    """Feature Pyramid Network to resize features maps from various backbone stages and concat them together to form
    a single high resolution feature map. Inspired from HrNet implementation in https://github.com/CASIA-IVA-Lab/ISP-reID/blob/master/modeling/backbones/cls_hrnet.py"""

    def __init__(self, input_dim, output_dim, mode='bilinear', img_size=None, spatial_scale=-1):
        super(MultiStageFusion, self).__init__()
        if spatial_scale > 0:
            self.spatial_size = np.array(img_size) / spatial_scale
        else:
            self.spatial_size = None
        self.mode = mode
        self.dim_reduce = BeforePoolingDimReduceLayer(input_dim, output_dim)

    def forward(self, features_per_stage):
        if self.spatial_size is None:
            spatial_size = features_per_stage[0].size()[2:]
        else:
            spatial_size = self.spatial_size
        resized_feature_maps = [features_per_stage[0]]
        for i in range(1, len(features_per_stage)):
            resized_feature_maps.append(F.interpolate(features_per_stage[i], size=spatial_size, mode=self.mode, align_corners=True))
        fused_features = torch.cat(resized_feature_maps, 1)
        fused_features = self.dim_reduce(fused_features)
        return fused_features


class PixelToPartClassifier(nn.Module):

    def __init__(self, dim_reduce_output, parts_num):
        super(PixelToPartClassifier, self).__init__()
        self.bn = torch.nn.BatchNorm2d(dim_reduce_output)
        self.classifier = nn.Conv2d(in_channels=dim_reduce_output, out_channels=parts_num + 1, kernel_size=1, stride=1, padding=0)
        self._init_params()

    def forward(self, x):
        x = self.bn(x)
        return self.classifier(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GlobalMaxPoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveMaxPool2d((1, 1))


class GlobalWeightedAveragePoolingHead(GlobalMaskWeightedPoolingHead):

    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)
        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = torch.sum(parts_features, dim=(-2, -1))
        part_masks_sum = torch.sum(part_masks.flatten(0, 1), dim=(-2, -1))
        part_masks_sum = torch.clamp(part_masks_sum, min=1e-06)
        parts_features_avg = torch.div(parts_features, part_masks_sum)
        parts_features = parts_features_avg.view(N, M, -1)
        return parts_features


def init_part_attention_pooling_head(normalization, pooling, dim_reduce_output):
    if pooling == 'gap':
        parts_attention_pooling_head = GlobalAveragePoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gmp':
        parts_attention_pooling_head = GlobalMaxPoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gwap':
        parts_attention_pooling_head = GlobalWeightedAveragePoolingHead(dim_reduce_output, normalization)
    else:
        raise ValueError('pooling type {} not supported'.format(pooling))
    return parts_attention_pooling_head


class KPR(nn.Module):
    """Keypoint Promptable Re-Identification model. This model is a re-implementation of the BPBReID model.
    """

    def __init__(self, num_classes, pretrained, loss, config, horizontal_stripes=False, **kwargs):
        super(KPR, self).__init__()
        self.model_cfg = config.model.kpr
        self.num_classes = num_classes
        self.parts_num = self.model_cfg.masks.parts_num
        self.horizontal_stripes = horizontal_stripes
        self.shared_parts_id_classifier = self.model_cfg.shared_parts_id_classifier
        self.test_use_target_segmentation = self.model_cfg.test_use_target_segmentation
        self.training_binary_visibility_score = self.model_cfg.training_binary_visibility_score
        self.testing_binary_visibility_score = self.model_cfg.testing_binary_visibility_score
        self.use_prompt_visibility_score = self.model_cfg.use_prompt_visibility_score
        self.enable_fpn = self.model_cfg.enable_fpn
        self.fpn_out_dim = self.model_cfg.fpn_out_dim
        self.enable_msf = self.model_cfg.enable_msf
        kwargs.pop('name', None)
        self.backbone_appearance_feature_extractor = models.build_model(self.model_cfg.backbone, num_classes, config=config, loss=loss, pretrained=pretrained, last_stride=self.model_cfg.last_stride, enable_dim_reduction=self.model_cfg.dim_reduce == 'before_pooling', dim_reduction_channels=self.model_cfg.dim_reduce_output, pretrained_path=config.model.backbone_pretrained_path, use_as_backbone=True, enable_fpn=self.enable_msf or self.enable_fpn, **kwargs)
        self.spatial_feature_shape = self.backbone_appearance_feature_extractor.spatial_feature_shape
        self.spatial_feature_depth = self.spatial_feature_shape[2]
        if self.enable_fpn:
            out_channels = self.fpn_out_dim if not self.enable_msf else int(self.fpn_out_dim / len(self.backbone_appearance_feature_extractor.spatial_feature_depth_per_layer))
            self.fpn = FeaturePyramidNetwork(self.backbone_appearance_feature_extractor.spatial_feature_depth_per_layer, out_channels=out_channels)
            self.spatial_feature_depth = out_channels
        if self.enable_msf:
            input_dim = self.fpn_out_dim if self.enable_fpn else self.backbone_appearance_feature_extractor.spatial_feature_depth_per_layer.sum()
            output_dim = self.backbone_appearance_feature_extractor.spatial_feature_depth_per_layer[-1]
            self.msf = MultiStageFusion(spatial_scale=self.model_cfg.msf_spatial_scale, img_size=(config.data.height, config.data.width), input_dim=input_dim, output_dim=output_dim)
            self.spatial_feature_depth = output_dim
        self.init_dim_reduce_layers(self.model_cfg.dim_reduce, self.spatial_feature_depth, self.model_cfg.dim_reduce_output)
        self.global_pooling_head = nn.AdaptiveAvgPool2d(1)
        self.foreground_attention_pooling_head = GlobalAveragePoolingHead(self.dim_reduce_output)
        self.background_attention_pooling_head = GlobalAveragePoolingHead(self.dim_reduce_output)
        self.parts_attention_pooling_head = init_part_attention_pooling_head(self.model_cfg.normalization, self.model_cfg.pooling, self.dim_reduce_output)
        self.learnable_attention_enabled = self.model_cfg.learnable_attention_enabled
        self.pixel_classifier = PixelToPartClassifier(self.spatial_feature_depth, self.parts_num)
        self.global_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.background_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.foreground_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.concat_parts_identity_classifier = BNClassifier(self.parts_num * self.dim_reduce_output, self.num_classes)
        if self.shared_parts_id_classifier:
            self.parts_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        else:
            self.parts_identity_classifier = nn.ModuleList([BNClassifier(self.dim_reduce_output, self.num_classes) for _ in range(self.parts_num)])

    def init_dim_reduce_layers(self, dim_reduce_mode, spatial_feature_depth, dim_reduce_output):
        self.dim_reduce_output = dim_reduce_output
        self.after_pooling_dim_reduce = False
        self.before_pooling_dim_reduce = None
        if dim_reduce_mode == 'before_pooling':
            self.before_pooling_dim_reduce = BeforePoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
            self.spatial_feature_depth = dim_reduce_output
        elif dim_reduce_mode == 'after_pooling':
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
        elif dim_reduce_mode == 'before_and_after_pooling':
            self.before_pooling_dim_reduce = BeforePoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output * 2)
            spatial_feature_depth = dim_reduce_output * 2
            self.spatial_feature_depth = spatial_feature_depth
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
        elif dim_reduce_mode == 'after_pooling_with_dropout':
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output, 0.5)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output, 0.5)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output, 0.5)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output, 0.5)
        else:
            self.dim_reduce_output = spatial_feature_depth

    def forward(self, images, target_masks=None, prompt_masks=None, keypoints_xyc=None, cam_label=None, **kwargs):
        """
        :param images: images tensor of size [N, C, Hi, Wi], where N is the batch size, C channel depth (3 for RGB), and
            (Hi, Wi) are the image height and width.
        :param target_masks: masks tensor of size [N, K+1, Hm, Wm], where N is the batch size, K is the number
            parts, and (Hm, Wm) are the masks height and width. The first index (index 0) along the parts K+1 dimension
            is the background by convention. The masks are expected to have values in the range [0, 1]. Spatial entry at
            location target_masks[i, k+1, h, w] is the probability that the pixel at location (h, w) belongs to
            part k for batch sample i. The masks are NOT expected to be of the same size as the images.
        :return:
        """
        spatial_features = self.backbone_appearance_feature_extractor(images, prompt_masks=prompt_masks, keypoints_xyc=keypoints_xyc, cam_label=cam_label)
        if self.enable_fpn and isinstance(spatial_features, dict):
            spatial_features = self.fpn(spatial_features)
        if isinstance(spatial_features, dict):
            if self.enable_msf:
                spatial_features = self.msf(spatial_features)
            else:
                spatial_features = spatial_features[0]
        N, _, Hf, Wf = spatial_features.shape
        if self.before_pooling_dim_reduce is not None and spatial_features.shape[1] != self.dim_reduce_output:
            spatial_features = self.before_pooling_dim_reduce(spatial_features)
        if self.horizontal_stripes:
            pixels_cls_scores = None
            feature_map_shape = Hf, Wf
            stripes_range = np.round(np.arange(0, self.parts_num + 1) * feature_map_shape[0] / self.parts_num).astype(int)
            pcb_masks = torch.zeros((self.parts_num, feature_map_shape[0], feature_map_shape[1]))
            for i in range(0, stripes_range.size - 1):
                pcb_masks[i, stripes_range[i]:stripes_range[i + 1], :] = 1
            pixels_parts_probabilities = pcb_masks
            pixels_parts_probabilities.requires_grad = False
        elif self.learnable_attention_enabled:
            pixels_cls_scores = self.pixel_classifier(spatial_features)
            pixels_parts_probabilities = F.softmax(pixels_cls_scores, dim=1)
        else:
            pixels_cls_scores = None
            assert target_masks is not None
            target_masks = target_masks.type(spatial_features.dtype)
            pixels_parts_probabilities = target_masks
            pixels_parts_probabilities.requires_grad = False
            assert pixels_parts_probabilities.max() <= 1 and pixels_parts_probabilities.min() >= 0
        background_masks = pixels_parts_probabilities[:, 0]
        parts_masks = pixels_parts_probabilities[:, 1:]
        if not self.training and self.test_use_target_segmentation == 'hard':
            assert target_masks is not None
            target_segmentation_mask = target_masks[:, 1:].max(dim=1)[0] > target_masks[:, 0]
            background_masks = ~target_segmentation_mask
            parts_masks[background_masks.unsqueeze(1).expand_as(parts_masks)] = 1e-12
        if not self.training and self.test_use_target_segmentation == 'soft':
            assert target_masks is not None
            parts_masks = parts_masks * target_masks[:, 1:]
        foreground_masks = parts_masks.max(dim=1)[0]
        global_masks = torch.ones_like(foreground_masks)
        if self.use_prompt_visibility_score:
            pixels_parts_probabilities = target_masks
        if self.training and self.training_binary_visibility_score or not self.training and self.testing_binary_visibility_score:
            pixels_parts_predictions = pixels_parts_probabilities.argmax(dim=1)
            pixels_parts_predictions_one_hot = F.one_hot(pixels_parts_predictions, self.parts_num + 1).permute(0, 3, 1, 2)
            parts_visibility = pixels_parts_predictions_one_hot.amax(dim=(2, 3))
        else:
            parts_visibility = pixels_parts_probabilities.amax(dim=(2, 3))
        background_visibility = parts_visibility[:, 0]
        foreground_visibility = parts_visibility.amax(dim=1)
        parts_visibility = parts_visibility[:, 1:]
        concat_parts_visibility = foreground_visibility
        global_visibility = torch.ones_like(foreground_visibility)
        global_embeddings = self.global_pooling_head(spatial_features).view(N, -1)
        foreground_embeddings = self.foreground_attention_pooling_head(spatial_features, foreground_masks.unsqueeze(1)).flatten(1, 2)
        background_embeddings = self.background_attention_pooling_head(spatial_features, background_masks.unsqueeze(1)).flatten(1, 2)
        parts_embeddings = self.parts_attention_pooling_head(spatial_features, parts_masks)
        if self.after_pooling_dim_reduce:
            global_embeddings = self.global_after_pooling_dim_reduce(global_embeddings)
            foreground_embeddings = self.foreground_after_pooling_dim_reduce(foreground_embeddings)
            background_embeddings = self.background_after_pooling_dim_reduce(background_embeddings)
            parts_embeddings = self.parts_after_pooling_dim_reduce(parts_embeddings)
        concat_parts_embeddings = parts_embeddings.flatten(1, 2)
        bn_global_embeddings, global_cls_score = self.global_identity_classifier(global_embeddings)
        bn_background_embeddings, background_cls_score = self.background_identity_classifier(background_embeddings)
        bn_foreground_embeddings, foreground_cls_score = self.foreground_identity_classifier(foreground_embeddings)
        bn_concat_parts_embeddings, concat_parts_cls_score = self.concat_parts_identity_classifier(concat_parts_embeddings)
        bn_parts_embeddings, parts_cls_score = self.parts_identity_classification(self.dim_reduce_output, N, parts_embeddings)
        embeddings = {GLOBAL: global_embeddings, BACKGROUND: background_embeddings, FOREGROUND: foreground_embeddings, CONCAT_PARTS: concat_parts_embeddings, PARTS: parts_embeddings, BN_GLOBAL: bn_global_embeddings, BN_BACKGROUND: bn_background_embeddings, BN_FOREGROUND: bn_foreground_embeddings, BN_CONCAT_PARTS: bn_concat_parts_embeddings, BN_PARTS: bn_parts_embeddings}
        visibility_scores = {GLOBAL: global_visibility, BACKGROUND: background_visibility, FOREGROUND: foreground_visibility, CONCAT_PARTS: concat_parts_visibility, PARTS: parts_visibility}
        id_cls_scores = {GLOBAL: global_cls_score, BACKGROUND: background_cls_score, FOREGROUND: foreground_cls_score, CONCAT_PARTS: concat_parts_cls_score, PARTS: parts_cls_score}
        masks = {GLOBAL: global_masks, BACKGROUND: background_masks, FOREGROUND: foreground_masks, CONCAT_PARTS: foreground_masks, PARTS: parts_masks}
        return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks

    def parts_identity_classification(self, D, N, parts_embeddings):
        if self.shared_parts_id_classifier:
            parts_embeddings = parts_embeddings.flatten(0, 1)
            bn_part_embeddings, part_cls_score = self.parts_identity_classifier(parts_embeddings)
            bn_part_embeddings = bn_part_embeddings.view([N, self.parts_num, D])
            part_cls_score = part_cls_score.view([N, self.parts_num, -1])
        else:
            scores = []
            embeddings = []
            for i, parts_identity_classifier in enumerate(self.parts_identity_classifier):
                bn_part_embeddings, part_cls_score = parts_identity_classifier(parts_embeddings[:, i])
                scores.append(part_cls_score.unsqueeze(1))
                embeddings.append(bn_part_embeddings.unsqueeze(1))
            part_cls_score = torch.cat(scores, 1)
            bn_part_embeddings = torch.cat(embeddings, 1)
        return bn_part_embeddings, part_cls_score


class MLFNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, fsm_channels, groups=32):
        super(MLFNBlock, self).__init__()
        self.groups = groups
        mid_channels = out_channels // 2
        self.fm_conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.fm_bn1 = nn.BatchNorm2d(mid_channels)
        self.fm_conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False, groups=self.groups)
        self.fm_bn2 = nn.BatchNorm2d(mid_channels)
        self.fm_conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.fm_bn3 = nn.BatchNorm2d(out_channels)
        self.fsm = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, fsm_channels[0], 1), nn.BatchNorm2d(fsm_channels[0]), nn.ReLU(inplace=True), nn.Conv2d(fsm_channels[0], fsm_channels[1], 1), nn.BatchNorm2d(fsm_channels[1]), nn.ReLU(inplace=True), nn.Conv2d(fsm_channels[1], self.groups, 1), nn.BatchNorm2d(self.groups), nn.Sigmoid())
        self.downsample = None
        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        s = self.fsm(x)
        x = self.fm_conv1(x)
        x = self.fm_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fm_conv2(x)
        x = self.fm_bn2(x)
        x = F.relu(x, inplace=True)
        b, c = x.size(0), x.size(1)
        n = c // self.groups
        ss = s.repeat(1, n, 1, 1)
        ss = ss.view(b, n, self.groups, 1, 1)
        ss = ss.permute(0, 2, 1, 3, 4).contiguous()
        ss = ss.view(b, c, 1, 1)
        x = ss * x
        x = self.fm_conv3(x)
        x = self.fm_bn3(x)
        x = F.relu(x, inplace=True)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return F.relu(residual + x, inplace=True), s


class MLFN(nn.Module):
    """Multi-Level Factorisation Net.

    Reference:
        Chang et al. Multi-Level Factorisation Net for
        Person Re-Identification. CVPR 2018.

    Public keys:
        - ``mlfn``: MLFN (Multi-Level Factorisation Net).
    """

    def __init__(self, num_classes, loss='softmax', groups=32, channels=[64, 256, 512, 1024, 2048], embed_dim=1024, **kwargs):
        super(MLFN, self).__init__()
        self.loss = loss
        self.groups = groups
        self.conv1 = nn.Conv2d(3, channels[0], 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.feature = nn.ModuleList([MLFNBlock(channels[0], channels[1], 1, [128, 64], self.groups), MLFNBlock(channels[1], channels[1], 1, [128, 64], self.groups), MLFNBlock(channels[1], channels[1], 1, [128, 64], self.groups), MLFNBlock(channels[1], channels[2], 2, [256, 128], self.groups), MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups), MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups), MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups), MLFNBlock(channels[2], channels[3], 2, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[4], 2, [512, 128], self.groups), MLFNBlock(channels[4], channels[4], 1, [512, 128], self.groups), MLFNBlock(channels[4], channels[4], 1, [512, 128], self.groups)])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_x = nn.Sequential(nn.Conv2d(channels[4], embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.fc_s = nn.Sequential(nn.Conv2d(self.groups * 16, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        s_hat = []
        for block in self.feature:
            x, s = block(x)
            s_hat.append(s)
        s_hat = torch.cat(s_hat, 1)
        x = self.global_avgpool(x)
        x = self.fc_x(x)
        s_hat = self.fc_s(s_hat)
        v = (x + s_hat) * 0.5
        v = v.view(v.size(0), -1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class MobileNetV2(nn.Module):
    """MobileNetV2.

    Reference:
        Sandler et al. MobileNetV2: Inverted Residuals and
        Linear Bottlenecks. CVPR 2018.

    Public keys:
        - ``mobilenetv2_x1_0``: MobileNetV2 x1.0.
        - ``mobilenetv2_x1_4``: MobileNetV2 x1.4.
    """

    def __init__(self, num_classes, width_mult=1, loss='softmax', fc_dims=None, dropout_p=None, **kwargs):
        super(MobileNetV2, self).__init__()
        self.loss = loss
        self.in_channels = int(32 * width_mult)
        self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)
        self.conv2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), 1, 1)
        self.conv3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), 2, 2)
        self.conv4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), 3, 2)
        self.conv5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), 4, 2)
        self.conv6 = self._make_layer(Bottleneck, 6, int(96 * width_mult), 3, 1)
        self.conv7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), 3, 2)
        self.conv8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), 1, 1)
        self.conv9 = ConvBlock(self.in_channels, self.feature_dim, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, self.feature_dim, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, t, c, n, s):
        layers = []
        layers.append(block(self.in_channels, c, t, s))
        self.in_channels = c
        for i in range(1, n):
            layers.append(block(self.in_channels, c, t))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class ConvLayers(nn.Module):
    """Preprocessing layers."""

    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv1 = ConvBlock(3, 48, k=3, s=1, p=1)
        self.conv2 = ConvBlock(48, 96, k=3, s=1, p=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        return x


class MultiScaleA(nn.Module):
    """Multi-scale stream layer A (Sec.3.1)"""

    def __init__(self):
        super(MultiScaleA, self).__init__()
        self.stream1 = nn.Sequential(ConvBlock(96, 96, k=1, s=1, p=0), ConvBlock(96, 24, k=3, s=1, p=1))
        self.stream2 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), ConvBlock(96, 24, k=1, s=1, p=0))
        self.stream3 = ConvBlock(96, 24, k=1, s=1, p=0)
        self.stream4 = nn.Sequential(ConvBlock(96, 16, k=1, s=1, p=0), ConvBlock(16, 24, k=3, s=1, p=1), ConvBlock(24, 24, k=3, s=1, p=1))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class Reduction(nn.Module):
    """Reduction layer (Sec.3.1)"""

    def __init__(self):
        super(Reduction, self).__init__()
        self.stream1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stream2 = ConvBlock(96, 96, k=3, s=2, p=1)
        self.stream3 = nn.Sequential(ConvBlock(96, 48, k=1, s=1, p=0), ConvBlock(48, 56, k=3, s=1, p=1), ConvBlock(56, 64, k=3, s=2, p=1))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y


class MultiScaleB(nn.Module):
    """Multi-scale stream layer B (Sec.3.1)"""

    def __init__(self):
        super(MultiScaleB, self).__init__()
        self.stream1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), ConvBlock(256, 256, k=1, s=1, p=0))
        self.stream2 = nn.Sequential(ConvBlock(256, 64, k=1, s=1, p=0), ConvBlock(64, 128, k=(1, 3), s=1, p=(0, 1)), ConvBlock(128, 256, k=(3, 1), s=1, p=(1, 0)))
        self.stream3 = ConvBlock(256, 256, k=1, s=1, p=0)
        self.stream4 = nn.Sequential(ConvBlock(256, 64, k=1, s=1, p=0), ConvBlock(64, 64, k=(1, 3), s=1, p=(0, 1)), ConvBlock(64, 128, k=(3, 1), s=1, p=(1, 0)), ConvBlock(128, 128, k=(1, 3), s=1, p=(0, 1)), ConvBlock(128, 256, k=(3, 1), s=1, p=(1, 0)))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        return s1, s2, s3, s4


class Fusion(nn.Module):
    """Saliency-based learning fusion layer (Sec.3.2)"""

    def __init__(self):
        super(Fusion, self).__init__()
        self.a1 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a2 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a3 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a4 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

    def forward(self, x1, x2, x3, x4):
        s1 = self.a1.expand_as(x1) * x1
        s2 = self.a2.expand_as(x2) * x2
        s3 = self.a3.expand_as(x3) * x3
        s4 = self.a4.expand_as(x4) * x4
        y = self.avgpool(s1 + s2 + s3 + s4)
        return y


class MuDeep(nn.Module):
    """Multiscale deep neural network.

    Reference:
        Qian et al. Multi-scale Deep Learning Architectures
        for Person Re-identification. ICCV 2017.

    Public keys:
        - ``mudeep``: Multiscale deep neural network.
    """

    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(MuDeep, self).__init__()
        self.loss = loss
        self.block1 = ConvLayers()
        self.block2 = MultiScaleA()
        self.block3 = Reduction()
        self.block4 = MultiScaleB()
        self.block5 = Fusion()
        self.fc = nn.Sequential(nn.Linear(256 * 16 * 8, 4096), nn.BatchNorm1d(4096), nn.ReLU())
        self.classifier = nn.Linear(4096, num_classes)
        self.feat_dim = 4096

    def featuremaps(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(*x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.classifier(x)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, x
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, name=None, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.name = name

    def forward(self, x):
        x = self.relu(x)
        if self.name == 'specific':
            x = nn.ZeroPad2d((1, 0, 1, 0))(x)
        x = self.separable_1(x)
        if self.name == 'specific':
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetAMobile(nn.Module):
    """Neural Architecture Search (NAS).

    Reference:
        Zoph et al. Learning Transferable Architectures
        for Scalable Image Recognition. CVPR 2018.

    Public keys:
        - ``nasnetamobile``: NASNet-A Mobile.
    """

    def __init__(self, num_classes, loss, stem_filters=32, penultimate_filters=1056, filters_multiplier=2, **kwargs):
        super(NASNetAMobile, self).__init__()
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        self.loss = loss
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters // 2, in_channels_right=2 * filters, out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters, out_channels_left=2 * filters, in_channels_right=6 * filters, out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=8 * filters, out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 * filters, out_channels_left=4 * filters, in_channels_right=12 * filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=16 * filters, out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(24 * filters, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_15 = self.relu(x_cell_15)
        x_cell_15 = F.avg_pool2d(x_cell_15, x_cell_15.size()[2:])
        x_cell_15 = x_cell_15.view(x_cell_15.size(0), -1)
        x_cell_15 = self.dropout(x_cell_15)
        return x_cell_15

    def forward(self, input):
        v = self.features(input)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError('Unknown gate activation: {}'.format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class LightConvStream(nn.Module):
    """Lightweight convolution stream."""

    def __init__(self, in_channels, out_channels, depth):
        super(LightConvStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(depth)
        layers = []
        layers += [LightConv3x3(in_channels, out_channels)]
        for i in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', conv1_IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3], dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, blocks, layer, in_channels, out_channels):
        layers = []
        layers += [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels)]
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class OSBlockINin(nn.Module):
    """Omni-scale feature learning block with instance normalization."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PCB(nn.Module):
    """Part-based Convolutional Baseline.

    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.

    Public keys:
        - ``pcb_p4``: PCB with 4-part strips.
        - ``pcb_p6``: PCB with 6-part strips.
    """

    def __init__(self, num_classes, loss, block, layers, parts=6, reduced_dim=256, nonlinear='relu', **kwargs):
        self.inplanes = 64
        super(PCB, self).__init__()
        self.loss = loss
        self.parts = parts
        self.feature_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.em = nn.ModuleList([self._construct_em_layer(reduced_dim, 512 * block.expansion) for _ in range(self.parts)])
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList([nn.Linear(self.feature_dim, num_classes, bias=False) for _ in range(self.parts)])
        self._init_params()

    def _construct_em_layer(self, fc_dims, input_dim, dropout_p=0.5):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        layers = []
        layers.append(nn.Conv2d(input_dim, fc_dims, 1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(fc_dims))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v_g = self.parts_avgpool(f)
        if not self.training:
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)
        y = []
        v = []
        for i in range(self.parts):
            v_g_i = v_g[:, :, i, :].view(v_g.size(0), -1, 1, 1)
            v_g_i = self.em[i](v_g_i)
            v_h_i = v_g_i.view(v_g_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)
            v.append(v_g_i)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            v_g = F.normalize(v_g, p=2, dim=1)
            return y, v_g.view(v_g.size(0), -1)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        None
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PromptableTransformerBackbone(nn.Module):
    """ class to be inherited by all promptable transformer backbones.
    It defines how prompt should be tokenized (i.e. the implementation of the prompt tokenizer).
    It also defines how camera information should be embedded similar to Transreid.
    """

    def __init__(self, patch_embed, masks_patch_embed, patch_embed_size, config, patch_embed_dim, feature_dim, use_negative_keypoints=False, camera=0, view=0, sie_xishu=1.0, masks_prompting=False, disable_inference_prompting=False, prompt_parts_num=0, **kwargs):
        super().__init__()
        self.feature_dim = self.num_features = feature_dim
        self.patch_embed_dim = patch_embed_dim
        self.masks_prompting = masks_prompting
        self.disable_inference_prompting = disable_inference_prompting
        self.prompt_parts_num = prompt_parts_num
        self.pose_encoding_strategy = config.pose_encoding_strategy
        self.pose_encoding_all_layers = config.pose_encoding_all_layers
        self.no_background_token = config.no_background_token
        self.use_negative_keypoints = use_negative_keypoints
        self.patch_embed_size = patch_embed_size
        self.patch_embed = patch_embed
        self.masks_patch_embed = masks_patch_embed
        self.num_patches = self.patch_embed_size[0] * self.patch_embed_size[1]
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        else:
            self.sie_embed = None
        self.num_part_tokens = self.prompt_parts_num + 1
        if self.use_negative_keypoints:
            self.num_part_tokens += 1
        self.parts_embed = nn.Parameter(torch.zeros(self.num_part_tokens, 1, self.patch_embed_dim))
        self.num_layers = 4
        if self.pose_encoding_all_layers:
            self.parts_embed_dim_upscales = nn.ModuleDict({str(self.patch_embed_dim * 2 ** i): AfterPoolingDimReduceLayer(self.patch_embed_dim, self.patch_embed_dim * 2 ** i) for i in range(self.num_layers - 1)})
        trunc_normal_(self.parts_embed, std=0.02)

    def _cam_embed(self, images, cam_label, view_label):
        reshape = False
        if len(images.shape) == 4:
            b, h, w, c = images.shape
            images = images.view(b, h * w, c)
            reshape = True
        if self.cam_num > 0 and self.view_num > 0:
            images = images + self.sie_xishu * self.sie_embed[cam_label * self.view_num + view_label]
        elif self.cam_num > 0:
            images = images + self.sie_xishu * self.sie_embed[cam_label]
        elif self.view_num > 0:
            images = images + self.sie_xishu * self.sie_embed[view_label]
        else:
            images = images
        if reshape:
            images = images.view(b, h, w, c)
        return images
    """The Prompt Tokenizer, to tokenize the input keypoint prompt information and add it to images tokens.
    Here, keypoints prompts in the (x, y, c) format are already pre-processed (see 'torchreid/data/datasets/dataset.py -> ImageDataset.getitem()') 
    and turned into dense heatmaps of shape (K+2, H, W) where K is the number of parts, and K+2 include the negative keypoints and the background, and H, W are the height and width of the image.
    'prompt_masks' is therefore a tensor of shape (B, K+2, H, W) where B is the batch size."""

    def _mask_embed(self, image_features, prompt_masks, input_size):
        if self.masks_prompting:
            if prompt_masks is not None and prompt_masks.shape[2:] != input_size:
                prompt_masks = F.interpolate(prompt_masks, size=input_size, mode='bilinear', align_corners=True)
            if self.disable_inference_prompting or prompt_masks is None:
                prompt_masks = torch.zeros([image_features.shape[0], self.num_part_tokens, input_size[0], input_size[1]], device=image_features.device)
                if not self.no_background_token:
                    prompt_masks[:, 0] = 1.0
            prompt_masks = prompt_masks.type(image_features.dtype)
            if self.pose_encoding_strategy == 'embed_heatmaps_patches':
                prompt_masks.requires_grad = False
                if self.no_background_token:
                    prompt_masks = prompt_masks[:, 1:]
                part_tokens = self.masks_patch_embed(prompt_masks)
                part_tokens = part_tokens[0] if isinstance(part_tokens, tuple) else part_tokens
            elif self.pose_encoding_strategy == 'spatialize_part_tokens':
                parts_embed = self.parts_embed
                if parts_embed.shape[-1] != image_features.shape[-1]:
                    parts_embed = self.parts_embed_dim_upscales[str(image_features.shape[-1])](parts_embed)
                prompt_masks.requires_grad = False
                parts_segmentation_map = prompt_masks.argmax(dim=1)
                part_tokens = parts_embed[parts_segmentation_map].squeeze(-2)
                if self.no_background_token:
                    part_tokens[parts_segmentation_map == 0] = 0
                if len(part_tokens.shape) != len(image_features.shape):
                    part_tokens = part_tokens.flatten(start_dim=1, end_dim=2)
            else:
                raise NotImplementedError
            image_features += part_tokens
        return image_features

    def _combine_layers(self, features, layers, prompt_masks):
        features_per_stage = OrderedDict()
        for i, layer in enumerate(layers):
            features_size = features.shape[-2:]
            features = layer(features)
            if self.pose_encoding_all_layers:
                self._mask_embed(features, prompt_masks, features_size)
            features_per_stage[i] = features.permute(0, 3, 1, 2)
        return features_per_stage


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


def _ntuple(n):

    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        return x
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        None
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def resize_pos_embed(posemb, posemb_new, hight, width):
    ntok_new = posemb_new.shape[1]
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))
    None
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


class ViT(nn.Module):
    """ Transformer-based Object Re-Identification
    """

    def __init__(self, config, use_negative_keypoints=False, img_size=224, patch_size=16, stride_size=16, in_chans=3, in_chans_masks=17, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu=1.0, masks_prompting=False, disable_inference_prompting=False, prompt_parts_num=0, **kwargs):
        super().__init__()
        self.masks_prompting = masks_prompting
        self.disable_inference_prompting = disable_inference_prompting
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.feature_dim = self.embed_dim
        self.local_feature = local_feature
        self.prompt_parts_num = prompt_parts_num
        self.pose_encoding_strategy = config.pose_encoding_strategy
        self.no_background_token = config.no_background_token
        self.use_negative_keypoints = use_negative_keypoints
        self.in_chans_masks = in_chans_masks
        if not self.no_background_token:
            self.in_chans_masks += 1
        if self.use_negative_keypoints:
            self.in_chans_masks += 1
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim)
            self.masks_patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=self.in_chans_masks, embed_dim=embed_dim)
        self.spatial_feature_shape = [self.patch_embed.num_y, self.patch_embed.num_x, self.embed_dim]
        self.img_size = img_size
        self.tkzd_img_size = self.patch_embed.num_y, self.patch_embed.num_x
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        self.parts_embed = nn.Parameter(torch.zeros(self.prompt_parts_num + 1, 1, embed_dim))
        None
        None
        None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.parts_embed, std=0.02)
        self.apply(self._init_weights)

    def forward(self, images, prompt_masks=None, keypoints_xyc=None, cam_label=None, view_label=None, **kwargs):
        images = self.patch_embed(images)
        images = self._mask_embed(images, prompt_masks)
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        images = torch.cat((cls_tokens, images), dim=1)
        images = self._pos_embed(images)
        images = self._cam_embed(images, cam_label, view_label)
        images = self.pos_drop(images)
        for blk in self.blocks:
            images = blk(images)
        images = self.norm(images)
        images = images[:, 1:, :].transpose(2, 1).unflatten(-1, self.tkzd_img_size)
        return images

    def _pos_embed(self, images):
        images = images + self.pos_embed
        return images

    def _cam_embed(self, images, cam_label, view_label):
        if self.cam_num > 0 and self.view_num > 0:
            images = images + self.sie_xishu * self.sie_embed[cam_label * self.view_num + view_label]
        elif self.cam_num > 0:
            images = images + self.sie_xishu * self.sie_embed[cam_label]
        elif self.view_num > 0:
            images = images + self.sie_xishu * self.sie_embed[view_label]
        else:
            images = images
        return images

    def _mask_embed(self, image_features, prompt_masks, input_size):
        if self.masks_prompting:
            if prompt_masks is not None and prompt_masks.shape[2:] != input_size:
                prompt_masks = F.interpolate(prompt_masks, size=input_size, mode='bilinear', align_corners=True)
            if self.disable_inference_prompting or prompt_masks is None:
                prompt_masks = torch.zeros([image_features.shape[0], self.num_part_tokens, input_size[0], input_size[1]], device=image_features.device)
                if not self.no_background_token:
                    prompt_masks[:, 0] = 1.0
            prompt_masks = prompt_masks.type(image_features.dtype)
            if self.pose_encoding_strategy == 'embed_heatmaps_patches':
                prompt_masks.requires_grad = False
                if self.no_background_token:
                    prompt_masks = prompt_masks[:, 1:]
                part_tokens = self.masks_patch_embed(prompt_masks)
                part_tokens = part_tokens[0] if isinstance(part_tokens, tuple) else part_tokens
            elif self.pose_encoding_strategy == 'spatialize_part_tokens':
                parts_embed = self.parts_embed
                if parts_embed.shape[-1] != image_features.shape[-1]:
                    parts_embed = self.parts_embed_dim_upscales[str(image_features.shape[-1])](parts_embed)
                prompt_masks.requires_grad = False
                parts_segmentation_map = prompt_masks.argmax(dim=1)
                part_tokens = parts_embed[parts_segmentation_map].squeeze(-2)
                if self.no_background_token:
                    part_tokens[parts_segmentation_map == 0] = 0
                if len(part_tokens.shape) != len(image_features.shape):
                    part_tokens = part_tokens.flatten(start_dim=1, end_dim=2)
            else:
                raise NotImplementedError
            image_features += part_tokens
        return image_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                if 'distilled' in model_path:
                    None
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                None
                None


class Conv1x1_att(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1_att, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class score_embedding(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels):
        super(score_embedding, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reg = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.reg(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Pose_Subnet(nn.Module):
    """
    PVP and PGA
    """

    def __init__(self, blocks, in_channels, channels, att_num=1, IN=False, matching_score_reg=False):
        super(Pose_Subnet, self).__init__()
        num_blocks = len(blocks)
        self.conv1 = ConvLayer(in_channels, channels[0], 7, stride=1, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], 1, channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(blocks[1], 1, channels[1], channels[2], reduce_spatial_size=False)
        self.conv4 = Conv3x3(channels[2], channels[2])
        self.conv_out = Conv1x1_att(channels[2], att_num)
        self.matching_score_reg = matching_score_reg
        if self.matching_score_reg:
            self.conv_score = score_embedding(channels[2], att_num)
        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN, gate_reduction=4))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN, gate_reduction=4))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_ = self.conv4(x)
        x = self.conv_out(x_)
        _, max_index = x.max(dim=1, keepdim=True)
        onehot_index = torch.zeros_like(x).scatter_(1, max_index, 1)
        if self.matching_score_reg:
            score = self.conv_score(x_)
            return x, score, onehot_index
        else:
            return x, onehot_index

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class pose_guide_att_Resnet(PCB):

    def __init__(self, num_classes, loss, block, layers, last_stride=2, parts=4, reduced_dim=None, nonlinear='relu', pose_inchannel=56, part_score_reg=False, **kwargs):
        super(pose_guide_att_Resnet, self).__init__(num_classes, loss, block, layers, last_stride=last_stride, parts=parts, reduced_dim=reduced_dim, nonlinear=nonlinear, **kwargs)
        self.part_score_reg = part_score_reg
        self.pose_subnet = Pose_Subnet(blocks=[OSBlock, OSBlock], in_channels=pose_inchannel, channels=[32, 32, 32], att_num=parts, matching_score_reg=part_score_reg)
        self.pose_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.parts_avgpool = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(self.parts)])

    def forward(self, x, pose_map):
        f = self.featuremaps(x)
        if self.part_score_reg:
            pose_att, part_score, onehot_index = self.pose_subnet(pose_map)
        else:
            pose_att, onehot_index = self.pose_subnet(pose_map)
        pose_att = pose_att * onehot_index
        pose_att_pool = self.pose_pool(pose_att)
        v_g = []
        for i in range(self.parts):
            v_g_i = f * pose_att[:, i, :, :].unsqueeze(1) / (pose_att_pool[:, i, :, :].unsqueeze(1) + 1e-06)
            v_g_i = self.parts_avgpool[i](v_g_i)
            v_g.append(v_g_i)
        if not self.training:
            v_g = torch.cat(v_g, dim=2)
            v_g = F.normalize(v_g, p=2, dim=1)
            if self.part_score_reg:
                return v_g.squeeze(), part_score
            else:
                return v_g.view(v_g.size(0), -1)
        y = []
        v = []
        for i in range(self.parts):
            v_g_i = self.em[i](v_g[i])
            v_h_i = v_g_i.view(v_g_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)
            v.append(v_g_i)
        if self.loss == 'softmax':
            if self.training:
                if self.part_score_reg:
                    return y, pose_att, part_score, v_g
                else:
                    return y, pose_att
            else:
                return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class ResNet(nn.Module):

    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class BatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            nn.init.constant_(self.weight, weight_init)
        if bias_init is not None:
            nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm, 'GN': lambda channels, **args: nn.GroupNorm(32, channels)}[norm]
    return norm(out_channels, **kwargs)


class Non_local(nn.Module):

    def __init__(self, in_channels, bn_norm, reduc_ratio=2):
        super(Non_local, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), get_norm(bn_norm, self.in_channels))
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class IBN(nn.Module):

    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, int(channel / reduction), bias=False), nn.ReLU(inplace=True), nn.Linear(int(channel / reduction), channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNetMid(nn.Module):
    """Residual network + mid-level features.
    
    Reference:
        Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
        Cross-Domain Instance Matching. arXiv:1711.08106.

    Public keys:
        - ``resnet50mid``: ResNet50 + mid-level feature fusion.
    """

    def __init__(self, num_classes, loss, block, layers, last_stride=2, fc_dims=None, **kwargs):
        self.inplanes = 64
        super(ResNetMid, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        assert fc_dims is not None
        self.fc_fusion = self._construct_fc_layer(fc_dims, 512 * block.expansion * 2)
        self.feature_dim += 512 * block.expansion
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x4a = self.layer4[0](x)
        x4b = self.layer4[1](x4a)
        x4c = self.layer4[2](x4b)
        return x4a, x4b, x4c

    def forward(self, x):
        x4a, x4b, x4c = self.featuremaps(x)
        v4a = self.global_avgpool(x4a)
        v4b = self.global_avgpool(x4b)
        v4c = self.global_avgpool(x4c)
        v4ab = torch.cat([v4a, v4b], 1)
        v4ab = v4ab.view(v4ab.size(0), -1)
        v4ab = self.fc_fusion(v4ab)
        v4c = v4c.view(v4c.size(0), -1)
        v = torch.cat([v4ab, v4c], 1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class SamReID(nn.Module):

    def __init__(self, sam_model: 'Sam') ->None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.feature_dim = 256

    def forward(self, images, masks=None, keypoints_xyc=None):
        """        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,"""
        features = self.model.image_encoder(images)
        point_labels = torch.ones_like(keypoints_xyc[:, :, -1])
        points = keypoints_xyc[:, :, :-1], point_labels
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, masks=None, boxes=None)
        low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=features, image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False)
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return low_res_masks


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """ResNeXt bottleneck type C with a Squeeze-and-Excitation module"""
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64.0)) * groups)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):
    """Squeeze-and-excitation network.
    
    Reference:
        Hu et al. Squeeze-and-Excitation Networks. CVPR 2018.

    Public keys:
        - ``senet154``: SENet154.
        - ``se_resnet50``: ResNet50 + SE.
        - ``se_resnet101``: ResNet101 + SE.
        - ``se_resnet152``: ResNet152 + SE.
        - ``se_resnext50_32x4d``: ResNeXt50 (groups=32, width=4) + SE.
        - ``se_resnext101_32x4d``: ResNeXt101 (groups=32, width=4) + SE.
        - ``se_resnet50_fc512``: (ResNet50 + SE) + FC.
    """

    def __init__(self, num_classes, loss, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, last_stride=2, fc_dims=None, **kwargs):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `classifier` layer.
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.loss = loss
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=last_stride, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class ChannelShuffle(nn.Module):

    def __init__(self, num_groups):
        super(ChannelShuffle, self).__init__()
        self.g = num_groups

    def forward(self, x):
        b, c, h, w = x.size()
        n = c // self.g
        x = x.view(b, self.g, n, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)
        return x


class ShuffleNet(nn.Module):
    """ShuffleNet.

    Reference:
        Zhang et al. ShuffleNet: An Extremely Efficient Convolutional Neural
        Network for Mobile Devices. CVPR 2018.

    Public keys:
        - ``shufflenet``: ShuffleNet (groups=3).
    """

    def __init__(self, num_classes, loss='softmax', num_groups=3, **kwargs):
        super(ShuffleNet, self).__init__()
        self.loss = loss
        self.conv1 = nn.Sequential(nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.stage2 = nn.Sequential(Bottleneck(24, cfg[num_groups][0], 2, num_groups, group_conv1x1=False), Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups), Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups), Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups))
        self.stage3 = nn.Sequential(Bottleneck(cfg[num_groups][0], cfg[num_groups][1], 2, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups))
        self.stage4 = nn.Sequential(Bottleneck(cfg[num_groups][1], cfg[num_groups][2], 2, num_groups), Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups), Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups), Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups))
        self.classifier = nn.Linear(cfg[num_groups][2], num_classes)
        self.feat_dim = cfg[num_groups][2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        if not self.training:
            return x
        y = self.classifier(x)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, x
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(inp), nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True), self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(branch_features), nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2.
    
    Reference:
        Ma et al. ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design. ECCV 2018.

    Public keys:
        - ``shufflenet_v2_x0_5``: ShuffleNetV2 x0.5.
        - ``shufflenet_v2_x1_0``: ShuffleNetV2 x1.0.
        - ``shufflenet_v2_x1_5``: ShuffleNetV2 x1.5.
        - ``shufflenet_v2_x2_0``: ShuffleNetV2 x2.0.
    """

    def __init__(self, num_classes, loss, stages_repeats, stages_out_channels, **kwargs):
        super(ShuffleNetV2, self).__init__()
        self.loss = loss
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_IBN(nn.Module):

    def __init__(self, last_stride, block, layers, frozen_stages=-1, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=last_stride)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            None
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function:

    .. math::
        \\text{GELU}(x) = x * \\Phi(x)
    where :math:`\\Phi(x)` is the Cumulative Distribution Function for
    Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        return F.gelu(input)


def build_activation_layer(act_cfg):
    if act_cfg['type'] == 'ReLU':
        act_layer = nn.ReLU(inplace=act_cfg['inplace'])
    elif act_cfg['type'] == 'GELU':
        act_layer = GELU()
    return act_layer


def build_dropout(drop_cfg):
    drop_layer = DropPath(drop_cfg['drop_prob'])
    return drop_layer


class FFN(BaseModule):

    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0, dropout_layer=None, add_identity=True, init_cfg=None, **kwargs):
        super(FFN, self).__init__()
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(Sequential(Linear(in_channels, feedforward_channels), self.activate, nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.
    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


def build_norm_layer(norm_cfg, embed_dims):
    assert norm_cfg['type'] == 'LN'
    norm_layer = nn.LayerNorm(embed_dims)
    return norm_cfg['type'], norm_layer


class PatchMerging(BaseModule):
    """Merge patch feature map.
    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.
    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified..
            Default: True.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=None, padding='corner', dilation=1, bias=False, norm_cfg=dict(type='LN'), init_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None
        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.
        Returns:
            tuple: Contains merged results and its spatial shape.
                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect input_size is `Sequence` but get {input_size}'
        H, W = input_size
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C).permute([0, 3, 1, 2])
        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]
        x = self.sampler(x)
        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1) // self.sampler.stride[1] + 1
        output_size = out_h, out_w
        x = x.transpose(1, 2)
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self, embed_dims, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop_rate=0.0, proj_drop_rate=0.0, init_cfg=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5
        self.init_cfg = init_cfg
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self, embed_dims, num_heads, window_size, shift_size=0, qkv_bias=True, qk_scale=None, attn_drop_rate=0, proj_drop_rate=0, dropout_layer=dict(type='DropPath', drop_prob=0.0), init_cfg=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size
        self.w_msa = WindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=to_2tuple(window_size), qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate, init_cfg=None)
        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None
        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size ** 2, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self, embed_dims, num_heads, feedforward_channels, window_size=7, shift=False, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), with_cp=False, init_cfg=None):
        super(SwinBlock, self).__init__()
        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2 if shift else 0, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate), init_cfg=None)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=2, ffn_drop=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate), act_cfg=act_cfg, add_identity=True, init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self, embed_dims, num_heads, feedforward_channels, depth, window_size=7, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, downsample=None, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), with_cp=False, init_cfg=None):
        super().__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(embed_dims=embed_dims, num_heads=num_heads, feedforward_channels=feedforward_channels, window_size=window_size, shift=False if i % 2 == 0 else True, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rates[i], act_cfg=act_cfg, norm_cfg=norm_cfg, with_cp=with_cp, init_cfg=None)
            self.blocks.append(block)
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def swin_converter(ckpt):
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k
        new_ckpt['backbone.' + new_k] = new_v
    return new_ckpt


def trunc_normal_init(module: 'nn.Module', mean: 'float'=0, std: 'float'=1, a: 'float'=-2, b: 'float'=2, bias: 'float'=0) ->None:
    if hasattr(module, 'weight') and module.weight is not None:
        _no_grad_trunc_normal_(module.weight, mean, std, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SwinTransformer(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030
    Inspiration from
    https://github.com/microsoft/Swin-Transformer
    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, pretrain_img_size=224, in_channels=3, embed_dims=96, patch_size=4, window_size=7, mlp_ratio=4, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), strides=(4, 2, 2, 2), out_indices=(0, 1, 2, 3), qkv_bias=True, qk_scale=None, patch_norm=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, use_abs_pos_embed=False, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), with_cp=False, pretrained=None, convert_weights=False, frozen_stages=-1, init_cfg=None, semantic_weight=0.0):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, f'The size of image should have length 1 or 2, but got {len(pretrain_img_size)}'
        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        super(SwinTransformer, self).__init__()
        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dims=embed_dims, conv_type='Conv2d', kernel_size=patch_size, stride=strides[0], norm_cfg=norm_cfg if patch_norm else None, init_cfg=None)
        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(torch.zeros((1, num_patches, embed_dims)))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(in_channels=in_channels, out_channels=2 * in_channels, stride=strides[i + 1], norm_cfg=norm_cfg if patch_norm else None, init_cfg=None)
            else:
                downsample = None
            stage = SwinBlockSequence(embed_dims=in_channels, num_heads=num_heads[i], feedforward_channels=mlp_ratio * in_channels, depth=depths[i], window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=downsample, act_cfg=act_cfg, norm_cfg=norm_cfg, with_cp=with_cp, init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.semantic_weight = semantic_weight
        if self.semantic_weight >= 0:
            self.semantic_embed_w = ModuleList()
            self.semantic_embed_b = ModuleList()
            for i in range(len(depths)):
                if i >= len(depths) - 1:
                    i = len(depths) - 2
                semantic_embed_w = nn.Linear(2, self.num_features[i + 1])
                semantic_embed_b = nn.Linear(2, self.num_features[i + 1])
                for param in semantic_embed_w.parameters():
                    param.requires_grad = False
                for param in semantic_embed_b.parameters():
                    param.requires_grad = False
                trunc_normal_init(semantic_embed_w, std=0.02, bias=0.0)
                trunc_normal_init(semantic_embed_b, std=0.02, bias=0.0)
                self.semantic_embed_w.append(semantic_embed_w)
                self.semantic_embed_b.append(semantic_embed_b)
            self.softplus = nn.Softplus()

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()
        for i in range(1, self.frozen_stages + 1):
            if i - 1 in self.out_indices:
                norm_layer = getattr(self, f'norm{i - 1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False
            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        logger = logging.getLogger('loading parameters.')
        if pretrained is None:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        else:
            ckpt = torch.load(pretrained, map_location='cpu')
            if 'teacher' in ckpt:
                ckpt = ckpt['teacher']
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                _state_dict = swin_converter(_state_dict)
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
            relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()
            res = self.load_state_dict(state_dict, False)
            None

    def forward(self, x, semantic_weight=None):
        if self.semantic_weight >= 0 and semantic_weight == None:
            w = torch.ones(x.shape[0], 1) * self.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.semantic_weight >= 0:
                sw = self.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.softplus(sw) + sb
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        x = self.avgpool(outs[-1])
        x = torch.flatten(x, 1)
        return x, outs


class GeneralizedMeanPooling(nn.Module):
    """Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-06):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool1d(x, self.output_size).pow(1.0 / self.p)


class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.local_feature = local_feature
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            None
            None
        None
        None
        None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            return x
        else:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x[:, 0]

    def forward(self, x, cam_label=None, view_label=None):
        x = self.forward_features(x, cam_label, view_label)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                if 'distilled' in model_path:
                    None
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                None
                None


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):

    def __init__(self, use_as_backbone, num_classes, camera_num, view_num, cfg, factory, model_filename):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = os.path.join(cfg.MODEL.PRETRAIN_PATH, model_filename[cfg.MODEL.TRANSFORMER_TYPE])
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.use_as_backbone = use_as_backbone
        None
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            None
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.spatial_feature_shape = [self.base.patch_embed.num_y, self.base.patch_embed.num_x, self.base.num_features]

    def forward(self, x, label=None, cam_label=None, view_label=None, **kwargs):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        feat = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(feat)
            embeddings = {GLOBAL: global_feat, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None, BN_GLOBAL: feat, BN_BACKGROUND: None, BN_FOREGROUND: None, BN_CONCAT_PARTS: None, BN_PARTS: None}
            visibility_scores = {GLOBAL: torch.ones(global_feat.shape[0], device=global_feat.device, dtype=torch.bool), BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            id_cls_scores = {GLOBAL: cls_score, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            masks = {GLOBAL: None, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            pixels_cls_scores = None
            spatial_features = None
            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks
        else:
            embeddings = {GLOBAL: global_feat, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None, BN_GLOBAL: feat, BN_BACKGROUND: None, BN_FOREGROUND: None, BN_CONCAT_PARTS: None, BN_PARTS: None}
            visibility_scores = {GLOBAL: torch.ones(global_feat.shape[0], device=global_feat.device, dtype=torch.bool), BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            id_cls_scores = {GLOBAL: None, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            masks = {GLOBAL: torch.ones((global_feat.shape[0], 32, 16), device=global_feat.device), BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            pixels_cls_scores = None
            spatial_features = None
            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        None

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        None


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(nn.Module):
    """SqueezeNet.

    Reference:
        Iandola et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
        and< 0.5 MB model size. arXiv:1602.07360.

    Public keys:
        - ``squeezenet1_0``: SqueezeNet (version=1.0).
        - ``squeezenet1_1``: SqueezeNet (version=1.1).
        - ``squeezenet1_0_fc512``: SqueezeNet (version=1.0) + FC.
    """

    def __init__(self, num_classes, loss, version=1.0, fc_dims=None, dropout_p=None, **kwargs):
        super(SqueezeNet, self).__init__()
        self.loss = loss
        self.feature_dim = 512
        if version not in [1.0, 1.1]:
            raise ValueError('Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'.format(version=version))
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(96, 16, 64, 64), Fire(128, 16, 64, 64), Fire(128, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 32, 128, 128), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(512, 64, 256, 256))
        else:
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.features(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)
    return x


class build_transformer_local(nn.Module):

    def __init__(self, use_as_backbone, num_classes, camera_num, view_num, cfg, factory, model_filename, rearrange):
        super(build_transformer_local, self).__init__()
        self.feature_dim = 768
        model_path = os.path.join(cfg.MODEL.PRETRAIN_PATH, model_filename[cfg.MODEL.TRANSFORMER_TYPE])
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.use_as_backbone = use_as_backbone
        None
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            None
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)
        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        None
        self.shift_num = cfg.MODEL.SHIFT_NUM
        None
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        None
        self.rearrange = rearrange
        self.spatial_feature_shape = [self.base.patch_embed.num_y, self.base.patch_embed.num_x, self.base.num_features]

    def forward(self, x, label=None, cam_label=None, view_label=None, **kwargs):
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        if self.use_as_backbone:
            return features[:, 1:, :].transpose(2, 1).unflatten(-1, (self.base.patch_embed.num_y, self.base.patch_embed.num_x))
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]
        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]
        b2_local_feat = x[:, patch_length:patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]
        b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]
        b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]
        feat = self.bottleneck(global_feat)
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)
        cls_score = self.classifier(feat)
        cls_score_1 = self.classifier_1(local_feat_1_bn)
        cls_score_2 = self.classifier_2(local_feat_2_bn)
        cls_score_3 = self.classifier_3(local_feat_3_bn)
        cls_score_4 = self.classifier_4(local_feat_4_bn)
        if self.training:
            embeddings = {GLOBAL: global_feat, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: torch.stack([local_feat_1, local_feat_2, local_feat_3, local_feat_4], dim=1), BN_GLOBAL: feat, BN_BACKGROUND: None, BN_FOREGROUND: None, BN_CONCAT_PARTS: None, BN_PARTS: torch.stack([local_feat_1_bn, local_feat_2_bn, local_feat_3_bn, local_feat_4_bn], dim=1)}
            visibility_scores = {GLOBAL: torch.ones(local_feat_1_bn.shape[0], device=local_feat_1_bn.device, dtype=torch.bool), BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: torch.ones((embeddings[PARTS].shape[0], embeddings[PARTS].shape[1]), device=local_feat_1_bn.device, dtype=torch.bool)}
            id_cls_scores = {GLOBAL: cls_score, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: torch.stack([cls_score_1, cls_score_2, cls_score_3, cls_score_4], dim=1)}
            masks = {GLOBAL: None, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            pixels_cls_scores = None
            spatial_features = None
            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks
        else:
            embeddings = {GLOBAL: torch.stack([global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1), BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None, BN_GLOBAL: feat, BN_BACKGROUND: None, BN_FOREGROUND: None, BN_CONCAT_PARTS: None, BN_PARTS: None}
            visibility_scores = {GLOBAL: torch.ones(local_feat_1_bn.shape[0], device=local_feat_1_bn.device, dtype=torch.bool), BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            id_cls_scores = {GLOBAL: None, BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            masks = {GLOBAL: torch.ones((local_feat_1_bn.shape[0], 32, 16), device=local_feat_1_bn.device), BACKGROUND: None, FOREGROUND: None, CONCAT_PARTS: None, PARTS: None}
            pixels_cls_scores = None
            spatial_features = None
            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        None

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        None


class Xception(nn.Module):
    """Xception.
    
    Reference:
        Chollet. Xception: Deep Learning with Depthwise
        Separable Convolutions. CVPR 2017.

    Public keys:
        - ``xception``: Xception.
    """

    def __init__(self, num_classes, loss, fc_dims=None, dropout_p=None, **kwargs):
        super(Xception, self).__init__()
        self.loss = loss
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 2048
        self.fc = self._construct_fc_layer(fc_dims, 2048, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptivePadding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AfterPoolingDimReduceLayer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (AvgPoolPad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BNClassifier,
     lambda: ([], {'in_dim': 4, 'class_num': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BeforePoolingDimReduceLayer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_filters': 4, 'out_filters': 4, 'reps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {}),
     True),
    (BranchSeparables,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BranchSeparablesReduction,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BranchSeparablesStem,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CellStem0,
     lambda: ([], {'stem_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CellStem1,
     lambda: ([], {'stem_filters': 4, 'num_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 8, 64, 64])], {}),
     False),
    (ChannelShuffle,
     lambda: ([], {'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1x1Linear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1x1_att,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'k': 4, 's': 4, 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLayers,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CountSketch,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseNet,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DimReduceLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'nonlinear': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneralizedMeanPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (GlobalAveragePoolingHead,
     lambda: ([], {'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalMaxPoolingHead,
     lambda: ([], {'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalWeightedAveragePoolingHead,
     lambda: ([], {'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardAttn,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IBN,
     lambda: ([], {'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetV2,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (InceptionV4,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (Inception_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Inception_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (Inception_C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LightConv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LightConvStream,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLFN,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MLFNBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 64, 'stride': 64, 'fsm_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPoolPad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mixed_3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Mixed_4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {}),
     True),
    (Mixed_5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiScaleA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 96, 64, 64])], {}),
     True),
    (MultiScaleB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (NASNetAMobile,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (NormalCell,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelToPartClassifier,
     lambda: ([], {'dim_reduce_output': 4, 'parts_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reduction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 96, 64, 64])], {}),
     True),
    (ReductionCell0,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReductionCell1,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reduction_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Reduction_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleNetV2,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss(), 'stages_repeats': [4, 4, 4], 'stages_out_channels': [4, 4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SqueezeNet,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (TripletLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (Xception,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (score_embedding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_VlSomers_keypoint_promptable_reidentification(_paritybench_base):
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

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

    def test_053(self):
        self._check(*TESTCASES[53])

    def test_054(self):
        self._check(*TESTCASES[54])

    def test_055(self):
        self._check(*TESTCASES[55])

    def test_056(self):
        self._check(*TESTCASES[56])

    def test_057(self):
        self._check(*TESTCASES[57])

    def test_058(self):
        self._check(*TESTCASES[58])

    def test_059(self):
        self._check(*TESTCASES[59])

    def test_060(self):
        self._check(*TESTCASES[60])

    def test_061(self):
        self._check(*TESTCASES[61])

    def test_062(self):
        self._check(*TESTCASES[62])

    def test_063(self):
        self._check(*TESTCASES[63])

    def test_064(self):
        self._check(*TESTCASES[64])

    def test_065(self):
        self._check(*TESTCASES[65])

    def test_066(self):
        self._check(*TESTCASES[66])

    def test_067(self):
        self._check(*TESTCASES[67])

    def test_068(self):
        self._check(*TESTCASES[68])

    def test_069(self):
        self._check(*TESTCASES[69])

    def test_070(self):
        self._check(*TESTCASES[70])

    def test_071(self):
        self._check(*TESTCASES[71])

    def test_072(self):
        self._check(*TESTCASES[72])

    def test_073(self):
        self._check(*TESTCASES[73])

    def test_074(self):
        self._check(*TESTCASES[74])

    def test_075(self):
        self._check(*TESTCASES[75])

    def test_076(self):
        self._check(*TESTCASES[76])

