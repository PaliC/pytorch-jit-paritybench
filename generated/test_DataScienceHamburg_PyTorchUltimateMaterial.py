import sys
_module = sys.modules[__name__]
del sys
dot_product_final = _module
dot_product_start = _module
nn_scratch_end = _module
nn_scratch_start = _module
Tensors = _module
MultiClassClassification_end = _module
MultiClassClassification_start = _module
MultilabelClassification_end = _module
MultilabelClassification_start = _module
CNN_BinaryClassification_end = _module
CNN_BinaryClassification_start = _module
ImagePreprocessing_end = _module
ImagePreprocessing_start = _module
LayerCalculations_end = _module
LayerCalculations_start = _module
Cnn_MulticlassClassification_end = _module
Cnn_MulticlassClassification_start = _module
DataPrep = _module
data_prep = _module
eda = _module
modeling = _module
plot_audio = _module
ObjectDetection = _module
yolo_data_prep = _module
perform_train = _module
StyleTransfer_end = _module
TransferLearning_end = _module
TransferLearning_start = _module
Flights_end = _module
Flights_start = _module
FunctionApproximation_end = _module
FunctionApproximation_incl_extrapolation_end = _module
FunctionApproximation_start = _module
MatrixFactorization_end = _module
MatrixFactorization_start = _module
Autoencoders_end = _module
Autoencoders_start = _module
GAN_Exercise_W03_Ch_224x224 = _module
GAN_Exercise_w01_Channels = _module
GAN_Exercise_w03_Channels = _module
Gan_end = _module
Image_Creation = _module
graph_intro_end = _module
node_classification_end = _module
node_classification_start = _module
ViT_custom = _module
lightning_intro_end = _module
lightning_intro_start = _module
semi_super_learn_end = _module
semi_super_learn_start = _module
super_learn = _module
chatgpt_examples = _module
cnn_hooks_end = _module
cnn_hooks_start = _module
elm = _module
image_sim_end = _module
image_sim_start = _module
inception = _module
app_gcp = _module
app_helloworld = _module
app_iris = _module
app_iris_weights_in_cloud = _module
model_class = _module
predGameOutcome = _module

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


import numpy as np


import pandas as pd


import torch.nn as nn


from sklearn.linear_model import LinearRegression


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from sklearn.model_selection import GridSearchCV


from sklearn.datasets import load_iris


from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score


from collections import Counter


from sklearn.datasets import make_multilabel_classification


import torchvision


import torchvision.transforms as transforms


import torch.nn.functional as F


import matplotlib.pyplot as plt


from torchvision import transforms


from typing import OrderedDict


from sklearn.metrics import confusion_matrix


from torch.optim import Adam


from torch.nn.functional import mse_loss


from torchvision import models


from collections import OrderedDict


from torch import optim


from torch import nn


from sklearn.preprocessing import MinMaxScaler


from sklearn import model_selection


from sklearn import preprocessing


from collections import defaultdict


from sklearn.metrics import mean_squared_error


from torchvision.datasets import ImageFolder


import torchvision.utils


import torch.optim as optim


import math


from random import uniform


from sklearn.manifold import TSNE


import random


from sklearn.feature_extraction.text import CountVectorizer


from torchvision.models import resnet18


import torchvision.models as models


from torch.autograd import Variable


class LinearRegressionTorch(nn.Module):

    def __init__(self, input_size=1, output_size=1):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class MultiClassNet(nn.Module):

    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x


class MultilabelNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MultilabelNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ImageClassificationNet(nn.Module):

    def __init__(self) ->None:
        pass

    def forward(self, x):
        pass


CLASSES = ['artifact', 'extrahls', 'murmur', 'normal']


NUM_CLASSES = len(CLASSES)


class ImageMulticlassClassificationNet(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 * 100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class FlightModel(nn.Module):

    def __init__(self, input_size=1, output_size=1):
        super(FlightModel, self).__init__()
        self.hidden_size = 50
        self.lstm = nn.LSTM(input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc1(torch.relu(output))
        return output


class TrigonometryModel(nn.Module):
    pass


class RecSysModel(nn.Module):

    def __init__(self, n_users, n_movies, n_embeddings=32):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, n_embeddings)
        self.movie_embed = nn.Embedding(n_movies, n_embeddings)
        self.out = nn.Linear(n_embeddings * 2, 1)

    def forward(self, users, movies):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        x = torch.cat([user_embeds, movie_embeds], dim=1)
        x = self.out(x)
        return x


LATENT_DIMS = 128


class Encoder(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 60 * 60, LATENT_DIMS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.fc = nn.Linear(LATENT_DIMS, 16 * 60 * 60)
        self.conv2 = nn.ConvTranspose2d(16, 6, 3)
        self.conv1 = nn.ConvTranspose2d(6, 1, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16, 60, 60)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


IN_FEATURES = 3 * 28 * 28


OUT_FEATURES = 1


class Discriminator(nn.Module):

    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(nn.Linear(in_features=IN_FEATURES, out_features=128), nn.LeakyReLU(negative_slope=0.1), nn.Linear(in_features=128, out_features=OUT_FEATURES), nn.Sigmoid())

    def forward(self, x):
        return self.disc(x)


img_dim = 3 * 28 * 28


class Generator(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.gen = nn.Sequential(nn.Linear(in_features=z_dim, out_features=256), nn.LeakyReLU(negative_slope=0.1), nn.Linear(in_features=256, out_features=img_dim))

    def forward(self, x):
        return self.gen(x)


class GCN(torch.nn.Module):

    def __init__(self, num_hidden, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):

    def __init__(self, num_hidden, num_features, num_classes, heads=8):
        super().__init__()
        self.conv1 = GATConv(num_features, num_hidden, heads)
        self.conv2 = GATConv(heads * num_hidden, num_classes, heads)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.3)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index)
        return x


class SesemiNet(nn.Module):

    def __init__(self, n_super_classes, n_selfsuper_classes) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out_super = nn.Linear(64, n_super_classes)
        self.fc_out_selfsuper = nn.Linear(64, n_selfsuper_classes)
        self.relu = nn.ReLU()
        self.output_layer_super = nn.Sigmoid()
        self.output_layer_selfsuper = nn.LogSoftmax()

    def backbone(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

    def forward(self, x_supervised, x_selfsupervised):
        x_supervised = self.backbone(x_supervised)
        x_supervised = self.fc_out_super(x_supervised)
        x_supervised = self.output_layer_super(x_supervised)
        x_selfsupervised = self.backbone(x_selfsupervised)
        x_selfsupervised = self.fc_out_selfsuper(x_selfsupervised)
        x_selfsupervised = self.output_layer_selfsuper(x_selfsupervised)
        return x_supervised, x_selfsupervised


class SupervisedNet(nn.Module):

    def __init__(self, n_super_classes) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out_super = nn.Linear(64, n_super_classes)
        self.relu = nn.ReLU()
        self.output_layer_super = nn.Sigmoid()

    def backbone(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_out_super(x)
        x = self.output_layer_super(x)
        return x


class SentimentModel(nn.Module):

    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN=10):
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x


class ImageClassficationInception(nn.Module):

    def __init__(self, in_channels=1, out_channels=4) ->None:
        super().__init__()
        self.branch1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, kernel_size=1), nn.BatchNorm2d(out_channels // 4), nn.ReLU())
        self.branch3x3 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, kernel_size=1), nn.BatchNorm2d(out_channels // 4), nn.ReLU(), nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels // 4), nn.ReLU())
        self.branch5x5 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, kernel_size=1), nn.BatchNorm2d(out_channels // 4), nn.ReLU(), nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2), nn.BatchNorm2d(out_channels // 4), nn.ReLU())
        self.branch_pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(in_channels, out_channels // 4, kernel_size=1), nn.BatchNorm2d(out_channels // 4), nn.ReLU())

    def forward(self, x):
        out1x1 = self.branch1x1(x)
        out3x3 = self.branch3x3(x)
        out5x5 = self.branch5x5(x)
        out_pool = self.branch_pool(x)
        out = torch.cat((out1x1, out3x3, out5x5, out_pool), dim=1)
        out = torch.flatten(out, 1)
        out = nn.Linear(out.shape[1], 1)(out)
        out = nn.Sigmoid()(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Autoencoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (FlightModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 1])], {}),
     True),
    (Generator,
     lambda: ([], {'z_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageClassficationInception,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (MultiClassNet,
     lambda: ([], {'NUM_FEATURES': 4, 'NUM_CLASSES': 4, 'HIDDEN_FEATURES': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultilabelNetwork,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SentimentModel,
     lambda: ([], {'NUM_FEATURES': 4, 'NUM_CLASSES': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_DataScienceHamburg_PyTorchUltimateMaterial(_paritybench_base):
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

