import sys
_module = sys.modules[__name__]
del sys
curobo_benchmark = _module
curobo_nvblox_benchmark = _module
curobo_nvblox_profile = _module
curobo_profile = _module
curobo_python_profile = _module
curobo_voxel_benchmark = _module
curobo_voxel_profile = _module
generate_nvblox_images = _module
ik_benchmark = _module
kinematics_benchmark = _module
metrics = _module
robometrics_benchmark = _module
collision_check_example = _module
ik_example = _module
batch_collision_checker = _module
batch_motion_gen_reacher = _module
collision_checker = _module
constrained_reacher = _module
helper = _module
ik_reachability = _module
load_all_robots = _module
motion_gen_reacher = _module
motion_gen_reacher_nvblox = _module
mpc_example = _module
mpc_nvblox_example = _module
multi_arm_reacher = _module
realsense_collision = _module
realsense_mpc = _module
realsense_reacher = _module
realsense_viewer = _module
simple_stacking = _module
convert_urdf_to_usd = _module
dowload_assets = _module
kinematics_example = _module
motion_gen_api_example = _module
motion_gen_example = _module
motion_gen_profile = _module
mpc_example = _module
nvblox_example = _module
pose_sequence_example = _module
robot_image_segmentation_example = _module
torch_layers_example = _module
trajopt_example = _module
usd_example = _module
world_representation_example = _module
setup = _module
curobo = _module
cuda_robot_model = _module
cuda_robot_generator = _module
cuda_robot_model = _module
kinematics_parser = _module
types = _module
urdf_kinematics_parser = _module
usd_kinematics_parser = _module
util = _module
curobolib = _module
geom = _module
kinematics = _module
ls = _module
opt = _module
tensor_step = _module
util_file = _module
cv = _module
sdf = _module
sdf_grid = _module
utils = _module
warp_primitives = _module
warp_sdf_fns = _module
warp_sdf_fns_deprecated = _module
world = _module
world_blox = _module
world_mesh = _module
world_voxel = _module
sphere_fit = _module
transform = _module
types = _module
graph = _module
graph_base = _module
graph_nx = _module
prm = _module
newton = _module
lbfgs = _module
newton_base = _module
opt_base = _module
particle = _module
parallel_es = _module
parallel_mppi = _module
particle_opt_base = _module
particle_opt_utils = _module
rollout = _module
arm_base = _module
arm_reacher = _module
cost = _module
bound_cost = _module
cost_base = _module
dist_cost = _module
manipulability_cost = _module
pose_cost = _module
primitive_collision_cost = _module
projected_dist_cost = _module
self_collision_cost = _module
stop_cost = _module
straight_line_cost = _module
zero_cost = _module
dynamics_model = _module
integration_utils = _module
kinematic_model = _module
model_base = _module
tensor_step = _module
rollout_base = _module
base = _module
camera = _module
enum = _module
file_path = _module
math = _module
robot = _module
state = _module
tensor = _module
error_metrics = _module
helpers = _module
logger = _module
sample_lib = _module
state_filter = _module
tensor_util = _module
torch_utils = _module
trajectory = _module
usd_helper = _module
warp = _module
warp_interpolation = _module
xrdf_utils = _module
wrap = _module
model = _module
curobo_robot_world = _module
robot_segmenter = _module
robot_world = _module
reacher = _module
evaluator = _module
ik_solver = _module
motion_gen = _module
mpc = _module
trajopt = _module
wrap_base = _module
wrap_mpc = _module
tests = _module
conftest = _module
cost_test = _module
cuda_graph_test = _module
cuda_robot_generator_test = _module
curobo_robot_world_model_test = _module
curobo_version_test = _module
geom_test = _module
geom_types_test = _module
goal_test = _module
ik_config_test = _module
ik_module_test = _module
ik_test = _module
interpolation_test = _module
kinematics_parser_test = _module
kinematics_test = _module
mimic_joint_test = _module
motion_gen_api_test = _module
motion_gen_constrained_test = _module
motion_gen_cuda_graph_test = _module
motion_gen_eval_test = _module
motion_gen_goalset_test = _module
motion_gen_js_test = _module
motion_gen_module_test = _module
motion_gen_speed_test = _module
motion_gen_test = _module
mpc_test = _module
multi_pose_test = _module
nvblox_test = _module
pose_reaching_test = _module
pose_test = _module
robot_assets_test = _module
robot_config_test = _module
robot_segmentation_test = _module
robot_world_model_test = _module
self_collision_test = _module
trajopt_config_test = _module
trajopt_test = _module
usd_export_test = _module
voxel_collision_test = _module
voxelization_test = _module
warp_mesh_test = _module
world_config_test = _module
xrdf_test = _module

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


from copy import deepcopy


from typing import Optional


import numpy as np


import torch


import time


import matplotlib.pyplot as plt


from typing import Any


from typing import Dict


from typing import List


import torch.autograd.profiler as profiler


from torch.profiler import ProfilerActivity


from torch.profiler import profile


from torch.profiler import record_function


from matplotlib import cm


import uuid


from torch import nn


from torch.utils.tensorboard import SummaryWriter


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import copy


from typing import Tuple


from typing import Union


from enum import Enum


from torch.autograd import Function


import math


import numpy


from typing import Sequence


from abc import abstractmethod


from itertools import product


from abc import abstractproperty


import scipy.interpolate as si


from scipy.stats.qmc import Halton


from torch.distributions.multivariate_normal import MultivariateNormal


from functools import lru_cache


class CuroboTorch(torch.nn.Module):

    def __init__(self, robot_world: 'RobotWorld'):
        """Build a simple structured NN:

        q_current -> kinematics -> sdf -> features
        [features, x_des] -> NN -> kinematics -> sdf -> [sdf, pose_distance] -> NN -> q_out
        loss = (fk(q_out) - x_des) + (q_current - q_out) + valid(q_out)
        """
        super(CuroboTorch, self).__init__()
        feature_dims = robot_world.kinematics.kinematics_config.link_spheres.shape[0] * 5 + 7 + 1
        q_feature_dims = 7
        final_feature_dims = feature_dims + 1 + 7
        output_dims = robot_world.kinematics.get_dof()
        self._robot_world = robot_world
        self._feature_mlp = nn.Sequential(nn.Linear(q_feature_dims, 512), nn.ReLU6(), nn.Linear(512, 512), nn.ReLU6(), nn.Linear(512, 512), nn.ReLU6(), nn.Linear(512, output_dims), nn.Tanh())
        self._final_mlp = nn.Sequential(nn.Linear(final_feature_dims, 256), nn.ReLU6(), nn.Linear(256, 256), nn.ReLU6(), nn.Linear(256, 64), nn.ReLU6(), nn.Linear(64, output_dims), nn.Tanh())

    def get_features(self, q: 'torch.Tensor', x_des: 'Optional[Pose]'=None):
        kin_state = self._robot_world.get_kinematics(q)
        spheres = kin_state.link_spheres_tensor.unsqueeze(2)
        q_sdf = self._robot_world.get_collision_distance(spheres)
        q_self = self._robot_world.get_self_collision_distance(kin_state.link_spheres_tensor.unsqueeze(1))
        features = [kin_state.link_spheres_tensor.view(q.shape[0], -1), q_sdf, q_self, kin_state.ee_position, kin_state.ee_quaternion]
        if x_des is not None:
            pose_distance = self._robot_world.pose_distance(x_des, kin_state.ee_pose, resize=True).view(-1, 1)
            features.append(pose_distance)
            features.append(x_des.position)
            features.append(x_des.quaternion)
        features = torch.cat(features, dim=-1)
        return features

    def forward(self, q: 'torch.Tensor', x_des: 'Pose'):
        """Forward for neural network

        Args:
            q (torch.Tensor): _description_
            x_des (torch.Tensor): _description_
        """
        in_features = torch.cat([x_des.position, x_des.quaternion], dim=-1)
        q_mid = self._feature_mlp(in_features)
        q_scale = self._robot_world.bound_scale * q_mid
        mid_features = self.get_features(q_scale, x_des=x_des)
        q_out = self._final_mlp(mid_features)
        q_out = self._robot_world.bound_scale * q_out
        return q_out

    def loss(self, x_des: 'Pose', q: 'torch.Tensor', q_in: 'torch.Tensor'):
        kin_state = self._robot_world.get_kinematics(q)
        distance = self._robot_world.pose_distance(x_des, kin_state.ee_pose, resize=True)
        d_sdf = self._robot_world.collision_constraint(kin_state.link_spheres_tensor.unsqueeze(1)).view(-1)
        d_self = self._robot_world.self_collision_cost(kin_state.link_spheres_tensor.unsqueeze(1)).view(-1)
        loss = 0.1 * torch.linalg.norm(q_in - q, dim=-1) + distance + 100.0 * (d_self + d_sdf)
        return loss

    def val_loss(self, x_des: 'Pose', q: 'torch.Tensor', q_in: 'torch.Tensor'):
        kin_state = self._robot_world.get_kinematics(q)
        distance = self._robot_world.pose_distance(x_des, kin_state.ee_pose, resize=True)
        d_sdf = self._robot_world.collision_constraint(kin_state.link_spheres_tensor.unsqueeze(1)).view(-1)
        d_self = self._robot_world.self_collision_cost(kin_state.link_spheres_tensor.unsqueeze(1)).view(-1)
        loss = 10.0 * (d_self + d_sdf) + distance
        return loss


class DistType(Enum):
    L1 = 0
    L2 = 1
    SQUARED_L2 = 2


class L2DistFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pos, target, target_idx, weight, run_weight, vec_weight, out_cost, out_cost_v, out_gp):
        wp_device = wp.device_from_torch(pos.device)
        b, h, dof = pos.shape
        requires_grad = pos.requires_grad
        wp.launch(kernel=forward_l2_warp, dim=b * h * dof, inputs=[wp.from_torch(pos.detach().reshape(-1), dtype=wp.float32), wp.from_torch(target.view(-1), dtype=wp.float32), wp.from_torch(target_idx.view(-1), dtype=wp.int32), wp.from_torch(weight, dtype=wp.float32), wp.from_torch(run_weight.view(-1), dtype=wp.float32), wp.from_torch(vec_weight.view(-1), dtype=wp.float32), wp.from_torch(out_cost_v.view(-1), dtype=wp.float32), wp.from_torch(out_gp.view(-1), dtype=wp.float32), requires_grad, b, h, dof], device=wp_device, stream=wp.stream_from_torch(pos.device))
        cost = torch.sum(out_cost_v, dim=-1)
        ctx.save_for_backward(out_gp)
        return cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        p_grad, = ctx.saved_tensors
        p_g = None
        if ctx.needs_input_grad[0]:
            p_g = p_grad
        return p_g, None, None, None, None, None, None, None, None


class L2DistLoopFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pos, target, target_idx, weight, run_weight, vec_weight, out_cost, out_cost_v, out_gp, l2_dof_kernel):
        wp_device = wp.device_from_torch(pos.device)
        b, h, dof = pos.shape
        wp.launch(kernel=l2_dof_kernel, dim=b * h, inputs=[wp.from_torch(pos.detach().view(-1).contiguous(), dtype=wp.float32), wp.from_torch(target.view(-1), dtype=wp.float32), wp.from_torch(target_idx.view(-1), dtype=wp.int32), wp.from_torch(weight, dtype=wp.float32), wp.from_torch(run_weight.view(-1), dtype=wp.float32), wp.from_torch(vec_weight.view(-1), dtype=wp.float32), wp.from_torch(out_cost.view(-1), dtype=wp.float32), wp.from_torch(out_gp.view(-1), dtype=wp.float32), pos.requires_grad, b, h, dof], device=wp_device, stream=wp.stream_from_torch(pos.device))
        ctx.save_for_backward(out_gp)
        return out_cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        p_grad, = ctx.saved_tensors
        p_g = None
        if ctx.needs_input_grad[0]:
            p_g = p_grad
        return p_g, None, None, None, None, None, None, None, None, None


def empty_decorator(function):
    return function


def log_info(txt: 'str', logger_name: 'str'='curobo', *args, **kwargs):
    """Log info message. Also see :py:meth:`logging.Logger.info`.

    Args:
        txt: Info message.
        logger_name: Name of the logger. Default is "curobo".
    """
    logger = logging.getLogger(logger_name)
    logger.info(txt, *args, **kwargs)


def get_torch_compile_options() ->dict:
    options = {}
    if is_torch_compile_available():
        from torch._inductor import config
        torch._dynamo.config.suppress_errors = True
        use_options = {'max_autotune': True, 'use_mixed_mm': True, 'conv_1x1_as_mm': True, 'coordinate_descent_tuning': True, 'epilogue_fusion': False, 'coordinate_descent_check_all_directions': True, 'force_fuse_int_mm_with_mul': True, 'triton.cudagraphs': False, 'aggressive_fusion': True, 'split_reductions': False, 'worker_start_method': 'spawn'}
        for k in use_options.keys():
            if hasattr(config, k):
                options[k] = use_options[k]
            else:
                log_info('Not found in torch.compile: ' + k)
    return options


def get_torch_jit_decorator(force_jit: 'bool'=False, dynamic: 'bool'=True, only_valid_for_compile: 'bool'=False):
    if not force_jit and is_torch_compile_available():
        return torch.compile(options=get_torch_compile_options(), dynamic=dynamic)
    elif not only_valid_for_compile:
        return torch.jit.script
    else:
        return empty_decorator


@dataclass(frozen=True)
class TensorDeviceType:
    device: 'torch.device' = torch.device('cuda', 0)
    dtype: 'torch.dtype' = torch.float32
    collision_geometry_dtype: 'torch.dtype' = torch.float32
    collision_gradient_dtype: 'torch.dtype' = torch.float32
    collision_distance_dtype: 'torch.dtype' = torch.float32

    @staticmethod
    def from_basic(device: 'str', dev_id: 'int'):
        return TensorDeviceType(torch.device(device, dev_id))

    def to_device(self, data_tensor):
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor
        else:
            return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor):
        return data_tensor

    def cpu(self):
        return TensorDeviceType(device=torch.device('cpu'), dtype=self.dtype)

    def as_torch_dict(self):
        return {'device': self.device, 'dtype': self.dtype}


def init_warp(quiet=True, tensor_args: 'TensorDeviceType'=TensorDeviceType()):
    wp.config.quiet = quiet
    wp.init()
    return True


def is_runtime_warp_kernel_enabled() ->bool:
    env_variable = os.environ.get('CUROBO_WARP_RUNTIME_KERNEL_DISABLE')
    if env_variable is None:
        return True
    return bool(int(env_variable))


def log_error(txt: 'str', logger_name: 'str'='curobo', exc_info=True, stack_info=False, stacklevel: 'int'=2, *args, **kwargs):
    """Log error and raise ValueError.

    Args:
        txt: Helpful message that conveys the error.
        logger_name: Name of the logger. Default is "curobo".
        exc_info: Add exception info to message. See :py:meth:`logging.Logger.error`.
        stack_info: Add stacktracke to message. See :py:meth:`logging.Logger.error`.
        stacklevel: See :py:meth:`logging.Logger.error`. Default value of 2 removes this function
            from the stack trace.

    Raises:
        ValueError: Error message with exception.
    """
    logger = logging.getLogger(logger_name)
    if sys.version_info.major == 3 and sys.version_info.minor <= 7:
        logger.error(txt, *args, exc_info=exc_info, stack_info=stack_info, **kwargs)
    else:
        logger.error(txt, *args, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, **kwargs)
    raise ValueError(txt)


def log_warn(txt: 'str', logger_name: 'str'='curobo', *args, **kwargs):
    """Log warning message. Also see :py:meth:`logging.Logger.warning`.

    Args:
        txt: Warning message.
        logger_name: Name of the logger. Default is "curobo".
    """
    logger = logging.getLogger(logger_name)
    logger.warning(txt, *args, **kwargs)


def is_lru_cache_avaiable():
    use_lru_cache = os.environ.get('CUROBO_USE_LRU_CACHE')
    if use_lru_cache is not None:
        return bool(int(use_lru_cache))
    log_info('Environment variable for CUROBO_USE_LRU_CACHE is not set, Enabling as default.')
    return False


def get_cache_fn_decorator(maxsize: 'Optional[int]'=None):
    if is_lru_cache_avaiable():
        return lru_cache(maxsize=maxsize)
    else:
        return empty_decorator


def warp_support_kernel_key(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version
    if version.parse(wp_version) < version.parse('1.2.1'):
        log_info('Warp version is ' + wp_version + ' < 1.2.1, using, creating global constant to trigger kernel generation.')
        return False
    return True


class OrientationError(Function):

    @staticmethod
    def geodesic_distance(goal_quat, current_quat, quat_res):
        quat_grad, rot_error = geodesic_distance(goal_quat, current_quat, quat_res)
        return quat_grad, rot_error

    @staticmethod
    def forward(ctx, goal_quat, current_quat, quat_res):
        quat_grad, rot_error = OrientationError.geodesic_distance(goal_quat, current_quat, quat_res)
        ctx.save_for_backward(quat_grad)
        return rot_error

    @staticmethod
    def backward(ctx, grad_out):
        grad_mul = grad_mul1 = None
        quat_grad, = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            grad_mul = grad_out * quat_grad
        if ctx.needs_input_grad[0]:
            grad_mul1 = -1.0 * grad_out * quat_grad
        return grad_mul1, grad_mul, None


class PoseErrorType(Enum):
    SINGLE_GOAL = 0
    BATCH_GOAL = 1
    GOALSET = 2
    BATCH_GOALSET = 3


def get_pose_distance(out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_q_vec, out_idx, current_position, goal_position, current_quat, goal_quat, vec_weight, weight, vec_convergence, run_weight, run_vec_weight, offset_waypoint, offset_tstep_fraction, batch_pose_idx, project_distance, batch_size, horizon, mode=1, num_goals=1, write_grad=False, write_distance=False, use_metric=False):
    if batch_pose_idx.shape[0] != batch_size:
        raise ValueError('Index buffer size is different from batch size')
    r = geom_cu.pose_distance(out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_q_vec, out_idx, current_position, goal_position.view(-1), current_quat, goal_quat.view(-1), vec_weight, weight, vec_convergence, run_weight, run_vec_weight, offset_waypoint, offset_tstep_fraction, batch_pose_idx, project_distance, batch_size, horizon, mode, num_goals, write_grad, write_distance, use_metric)
    out_distance = r[0]
    out_position_distance = r[1]
    out_rotation_distance = r[2]
    out_p_vec = r[3]
    out_q_vec = r[4]
    out_idx = r[5]
    return out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_q_vec, out_idx


class PoseError(torch.autograd.Function):

    @staticmethod
    def forward(ctx, current_position: 'torch.Tensor', goal_position: 'torch.Tensor', current_quat: 'torch.Tensor', goal_quat, vec_weight, weight, vec_convergence, run_weight, run_vec_weight, offset_waypoint, offset_tstep_fraction, batch_pose_idx, project_distance, out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_r_vec, out_idx, out_p_grad, out_q_grad, batch_size, horizon, mode, num_goals, use_metric, return_loss):
        """Compute error in pose

        _extended_summary_

        Args:
            ctx: _description_
            current_position: _description_
            goal_position: _description_
            current_quat: _description_
            goal_quat: _description_
            vec_weight: _description_
            weight: _description_
            vec_convergence: _description_
            run_weight: _description_
            run_vec_weight: _description_
            offset_waypoint: _description_
            offset_tstep_fraction: _description_
            batch_pose_idx: _description_
            out_distance: _description_
            out_position_distance: _description_
            out_rotation_distance: _description_
            out_p_vec: _description_
            out_r_vec: _description_
            out_idx: _description_
            out_p_grad: _description_
            out_q_grad: _description_
            batch_size: _description_
            horizon: _description_
            mode: _description_
            num_goals: _description_
            use_metric: _description_
            project_distance: _description_
            return_loss: _description_

        Returns:
            _description_
        """
        ctx.return_loss = return_loss
        out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_r_vec, out_idx = get_pose_distance(out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_r_vec, out_idx, current_position.contiguous(), goal_position, current_quat.contiguous(), goal_quat, vec_weight, weight, vec_convergence, run_weight, run_vec_weight, offset_waypoint, offset_tstep_fraction, batch_pose_idx, project_distance, batch_size, horizon, mode, num_goals, current_position.requires_grad, False, use_metric)
        ctx.save_for_backward(out_p_vec, out_r_vec)
        return out_distance

    @staticmethod
    def backward(ctx, grad_out_distance):
        pos_grad = None
        quat_grad = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            g_vec_p, g_vec_q = ctx.saved_tensors
            pos_grad = g_vec_p
            quat_grad = g_vec_q
            if ctx.return_loss:
                pos_grad = pos_grad * grad_out_distance.unsqueeze(1)
                quat_grad = quat_grad * grad_out_distance.unsqueeze(1)
        elif ctx.needs_input_grad[0]:
            g_vec_p, g_vec_q = ctx.saved_tensors
            pos_grad = g_vec_p
            if ctx.return_loss:
                pos_grad = pos_grad * grad_out_distance.unsqueeze(1)
        elif ctx.needs_input_grad[2]:
            g_vec_p, g_vec_q = ctx.saved_tensors
            quat_grad = g_vec_q
            if ctx.return_loss:
                quat_grad = quat_grad * grad_out_distance.unsqueeze(1)
        return pos_grad, None, quat_grad, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def get_pose_distance_backward(out_grad_p, out_grad_q, grad_distance, grad_p_distance, grad_q_distance, pose_weight, grad_p_vec, grad_q_vec, batch_size, use_distance=False):
    r = geom_cu.pose_distance_backward(out_grad_p, out_grad_q, grad_distance, grad_p_distance, grad_q_distance, pose_weight, grad_p_vec, grad_q_vec, batch_size, use_distance)
    return r[0], r[1]


class PoseErrorDistance(torch.autograd.Function):

    @staticmethod
    def forward(ctx, current_position, goal_position, current_quat, goal_quat, vec_weight, weight, vec_convergence, run_weight, run_vec_weight, offset_waypoint, offset_tstep_fraction, batch_pose_idx, project_distance, out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_r_vec, out_idx, out_p_grad, out_q_grad, batch_size, horizon, mode, num_goals, use_metric):
        out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_r_vec, out_idx = get_pose_distance(out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_r_vec, out_idx, current_position.contiguous(), goal_position, current_quat.contiguous(), goal_quat, vec_weight, weight, vec_convergence, run_weight, run_vec_weight, offset_waypoint, offset_tstep_fraction, batch_pose_idx, project_distance, batch_size, horizon, mode, num_goals, current_position.requires_grad, True, use_metric)
        ctx.save_for_backward(out_p_vec, out_r_vec, weight, out_p_grad, out_q_grad)
        return out_distance, out_position_distance, out_rotation_distance, out_idx

    @staticmethod
    def backward(ctx, grad_out_distance, grad_g_dist, grad_r_err, grad_out_idx):
        g_vec_p, g_vec_q, weight, out_grad_p, out_grad_q = ctx.saved_tensors
        pos_grad = None
        quat_grad = None
        batch_size = g_vec_p.shape[0] * g_vec_p.shape[1]
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            pos_grad, quat_grad = get_pose_distance_backward(out_grad_p, out_grad_q, grad_out_distance.contiguous(), grad_g_dist.contiguous(), grad_r_err.contiguous(), weight, g_vec_p, g_vec_q, batch_size, use_distance=True)
        elif ctx.needs_input_grad[0]:
            pos_grad = backward_PoseError_jit(grad_g_dist, grad_out_distance, weight[1], g_vec_p)
        elif ctx.needs_input_grad[2]:
            quat_grad = backward_PoseError_jit(grad_r_err, grad_out_distance, weight[0], g_vec_q)
        return pos_grad, None, quat_grad, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def interpolate_kernel(h, int_steps, tensor_args: 'TensorDeviceType'):
    mat = torch.zeros(((h - 1) * int_steps, h), device=tensor_args.device, dtype=tensor_args.dtype)
    delta = torch.arange(0, int_steps, device=tensor_args.device, dtype=tensor_args.dtype) / (int_steps - 1)
    for i in range(h - 1):
        mat[i * int_steps:i * int_steps + int_steps, i] = delta.flip(0)
        mat[i * int_steps:i * int_steps + int_steps, i + 1] = delta
    return mat


def sum_matrix(h, int_steps, tensor_args):
    sum_mat = torch.zeros(((h - 1) * int_steps, h), **tensor_args.as_torch_dict())
    for i in range(h - 1):
        sum_mat[i * int_steps:i * int_steps + int_steps, i] = 1.0
    return sum_mat


class ProjType(Enum):
    IDENTITY = 0
    PSEUDO_INVERSE = 1


def get_self_collision_distance(out_distance, out_vec, sparse_index, robot_spheres, collision_offset, weight, coll_matrix, thread_locations, thread_size, b_size, nspheres, compute_grad, checks_per_thread=32, experimental_kernel=True):
    r = geom_cu.self_collision_distance(out_distance, out_vec, sparse_index, robot_spheres, collision_offset, weight, coll_matrix, thread_locations, thread_size, b_size, nspheres, compute_grad, checks_per_thread, experimental_kernel)
    out_distance = r[0]
    out_vec = r[1]
    return out_distance, out_vec


class SelfCollisionDistance(torch.autograd.Function):

    @staticmethod
    def forward(ctx, out_distance, out_vec, sparse_idx, robot_spheres, sphere_offset, weight, coll_matrix, thread_locations, max_thread, checks_per_thread: 'int', experimental_kernel: 'bool', return_loss: 'bool'=False):
        b, h, n_spheres, _ = robot_spheres.shape
        out_distance, out_vec = get_self_collision_distance(out_distance, out_vec, sparse_idx, robot_spheres, sphere_offset, weight, coll_matrix.view(-1), thread_locations, max_thread, b * h, n_spheres, robot_spheres.requires_grad, checks_per_thread, experimental_kernel)
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_vec)
        return out_distance

    @staticmethod
    def backward(ctx, grad_out_distance):
        sphere_grad = None
        if ctx.needs_input_grad[3]:
            g_vec, = ctx.saved_tensors
            if ctx.return_loss:
                g_vec = g_vec * grad_out_distance.unsqueeze(1)
            sphere_grad = g_vec
        return None, None, None, sphere_grad, None, None, None, None, None, None, None, None


class RunSquaredSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cost_vec, weight, run_weight):
        cost = run_squared_sum(cost_vec, weight, run_weight)
        ctx.save_for_backward(cost_vec, weight, run_weight)
        return cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        cost_vec, w, r_w = ctx.saved_tensors
        c_grad = None
        if ctx.needs_input_grad[0]:
            c_grad = backward_run_squared_sum(cost_vec, w, r_w)
        return c_grad, None, None


class SquaredSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cost_vec, weight):
        cost = squared_sum(cost_vec, weight)
        ctx.save_for_backward(cost_vec, weight)
        return cost

    @staticmethod
    def backward(ctx, grad_out_cost):
        cost_vec, w = ctx.saved_tensors
        c_grad = None
        if ctx.needs_input_grad[0]:
            c_grad = backward_squared_sum(cost_vec, w)
        return c_grad, None

