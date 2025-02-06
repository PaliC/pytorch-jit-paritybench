import sys
_module = sys.modules[__name__]
del sys
Poisson_Neumann = _module
NSCoupled = _module
NSfracStep = _module
NSfracStep_ALE_stenosis_tubefinal = _module
NSfracStep_rotateinlet = _module
oasis_2017_Amir_vs = _module
common = _module
io = _module
utilities = _module
oasis = _module
Cylinder = _module
DrivenCavity = _module
Nozzle2D = _module
Skewed2D = _module
SkewedFlow = _module
AAA_BC = _module
AAA_BC_P16 = _module
AAA_monsoon = _module
AAA_monsoon2 = _module
AAA_monsoon_RT = _module
AAA_monsoon_RTtest = _module
AAA_monsoon_RTthresh = _module
AAA_monsoon_RTthresh10 = _module
AAA_monsoon_RTthresh1p5 = _module
AAA_monsoon_finalmesh = _module
AAA_monsoon_finalmeshRT = _module
AAA_monsoon_finalmesh_O1 = _module
AAA_monsoon_finalmeshnonNewt = _module
AAA_monsoon_finalmeshnonNewt_O1 = _module
Channel = _module
DrivenCavity3D = _module
FlowPastSphere3D = _module
IA05 = _module
IA05_BC = _module
IA05_O1 = _module
IA05_O1_12cyc = _module
IA05_O1_12cyc_tol = _module
IA05_O1_RT10 = _module
IA05_O1_RT3 = _module
IA05_O1_tol = _module
IA05_O2 = _module
IA05_O2RT1p5_12cyc = _module
IA05_O2RT3_12cyc = _module
IA05_O2_12cyc = _module
IA05_O2_tol = _module
IA05_O2nonNewt_12cyc = _module
IA05_O2quadrule = _module
IA05_RT1p5 = _module
IA05_RT3 = _module
IA05_RT3_12cyc = _module
IA05_RT3_tol = _module
IA05_RT5 = _module
IA05_nonNewt = _module
LaminarChannel = _module
Lshape = _module
TaylorGreen2D = _module
TaylorGreen3D = _module
coronary_image = _module
coronary_tube = _module
coronary_tube_meshindep = _module
coronary_tubefinal = _module
AAA = _module
AAA2 = _module
tube = _module
tube_monsoon = _module
problems = _module
cylindrical = _module
default = _module
naive = _module
BDFPC = _module
BDFPC_Fast = _module
Chorin = _module
IPCS = _module
IPCS_ABCN = _module
IPCS_ABCN_1outlet = _module
IPCS_ABCN_1outlet_memory = _module
IPCS_ABE = _module
DynamicLagrangian = _module
DynamicModules = _module
KineticEnergySGS = _module
NoModel = _module
ScaleDepDynamicLagrangian = _module
Smagorinsky = _module
Wale = _module
LES = _module
solvers = _module
Darcy_generatedata = _module
stenosis_steady_NS = _module
aneurysm_unsteady_NS = _module
inlet_data = _module
adv_diff_chaotic = _module
FENICS_import_mesh = _module
hdf5_utilities = _module
vtk2XML = _module
superresolution_image2image = _module
torch2mat = _module
torch2vtk = _module
VTK_normals_crop_process = _module
VTK_grad_curl = _module
Interpolate_velocity_newmesh = _module
getWSS_div = _module
Seed_surface_tracers = _module
deposition_concentration = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torchvision.transforms as transforms


import torchvision.datasets as dsets


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.utils.data import RandomSampler


import numpy as np


from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt


from matplotlib import pyplot as plt


from torch.autograd import Variable


import torch.optim as optim


from math import exp


from math import sqrt


from math import pi


import time


import math


from torch import nn


from scipy.io import savemat


class myCNN(nn.Module):

    def __init__(self):
        super(myCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.cnn11 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
        self.relu11 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(288, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 28 * 28)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.cnn11(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.tconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.tconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.tconv1(x))
        x = F.sigmoid(self.tconv2(x))
        return x


class DeepAutoencoder_original(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 28 * 28))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAutoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(32, 44), torch.nn.ReLU(), torch.nn.Linear(44, 64), torch.nn.ReLU(), torch.nn.Linear(64, 80), torch.nn.ReLU(), torch.nn.Linear(80, 28 * 28))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAutoencoder_deeper(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 28 * 28))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (autoencoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_amir_cardiolab_Py4SciComp(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

