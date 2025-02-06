# AOT ID: ['7_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: inductor_cache/wz/cwzsstayl5kqirvnalj2lkwwkns44cfpl2ln3idqjrrq67uwin3k.py
# Topologically Sorted Source Nodes: [l1_loss, mul, mul_1, loss_G_GAN_Feat, l1_loss_1, mul_2, mul_3, loss_G_GAN_Feat_1, l1_loss_2, mul_4, mul_5, loss_G_GAN_Feat_2, l1_loss_3, mul_6, mul_7, loss_G_GAN_Feat_3, l1_loss_4, mul_8, mul_9, loss_G_GAN_Feat_4, l1_loss_5, mul_10, mul_11, loss_G_GAN_Feat_5, l1_loss_6, mul_12, mul_13, loss_G_GAN_Feat_6, l1_loss_7, mul_14, mul_15, loss_G_GAN_Feat_7, l1_loss_8, mul_16, mul_17, loss_G_GAN_Feat_8, l1_loss_9, mul_18, mul_19, loss_G_GAN_Feat_9, l1_loss_10, mul_20, mul_21, loss_G_GAN_Feat_10, l1_loss_11, mul_22, mul_23, loss_G_GAN_Feat_11], Original ATen: [aten.sub, aten.abs, aten.mean, aten.mul, aten.add]
# Source node to ATen node mapping:
#   l1_loss => abs_1, mean, sub
#   l1_loss_1 => abs_2, mean_1, sub_1
#   l1_loss_10 => abs_11, mean_10, sub_10
#   l1_loss_11 => abs_12, mean_11, sub_11
#   l1_loss_2 => abs_3, mean_2, sub_2
#   l1_loss_3 => abs_4, mean_3, sub_3
#   l1_loss_4 => abs_5, mean_4, sub_4
#   l1_loss_5 => abs_6, mean_5, sub_5
#   l1_loss_6 => abs_7, mean_6, sub_6
#   l1_loss_7 => abs_8, mean_7, sub_7
#   l1_loss_8 => abs_9, mean_8, sub_8
#   l1_loss_9 => abs_10, mean_9, sub_9
#   loss_G_GAN_Feat => add
#   loss_G_GAN_Feat_1 => add_1
#   loss_G_GAN_Feat_10 => add_10
#   loss_G_GAN_Feat_11 => add_11
#   loss_G_GAN_Feat_2 => add_2
#   loss_G_GAN_Feat_3 => add_3
#   loss_G_GAN_Feat_4 => add_4
#   loss_G_GAN_Feat_5 => add_5
#   loss_G_GAN_Feat_6 => add_6
#   loss_G_GAN_Feat_7 => add_7
#   loss_G_GAN_Feat_8 => add_8
#   loss_G_GAN_Feat_9 => add_9
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_1, %select_3), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, 0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_5, %select_7), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_1, 0.5), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, 4), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_9, %select_11), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_2,), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_3,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_2, 0.5), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, 4), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_5), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_13, %select_15), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_3,), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_4,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_3, 0.5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, 4), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_7), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_17, %select_19), kwargs = {})
#   %abs_5 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_4,), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_5,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_4, 0.5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, 4), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_9), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_21, %select_23), kwargs = {})
#   %abs_6 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_5,), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_6,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_5, 0.5), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 4), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_11), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_25, %select_27), kwargs = {})
#   %abs_7 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_6,), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_7,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_6, 0.5), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, 4), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_13), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_29, %select_31), kwargs = {})
#   %abs_8 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_7,), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_8,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_7, 0.5), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, 4), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_15), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_33, %select_35), kwargs = {})
#   %abs_9 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_8,), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_9,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_8, 0.5), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 4), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mul_17), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_37, %select_39), kwargs = {})
#   %abs_10 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_9,), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_10,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_9, 0.5), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, 4), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mul_19), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_41, %select_43), kwargs = {})
#   %abs_11 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_10,), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_11,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_10, 0.5), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, 4), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %mul_21), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_45, %select_47), kwargs = {})
#   %abs_12 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_11,), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%abs_12,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_11, 0.5), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, 4), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_23), kwargs = {})
triton_per_fused_abs_add_mean_mul_sub_0 = async_compile.triton('triton_per_fused_abs_add_mean_mul_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_mean_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 12, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_mean_mul_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp7 = tl.load(in_ptr0 + (16 + r0), None)
    tmp8 = tl.load(in_ptr1 + (16 + r0), None)
    tmp14 = tl.load(in_ptr0 + (32 + r0), None)
    tmp15 = tl.load(in_ptr1 + (32 + r0), None)
    tmp21 = tl.load(in_ptr0 + (64 + r0), None)
    tmp22 = tl.load(in_ptr1 + (64 + r0), None)
    tmp28 = tl.load(in_ptr0 + (80 + r0), None)
    tmp29 = tl.load(in_ptr1 + (80 + r0), None)
    tmp35 = tl.load(in_ptr0 + (96 + r0), None)
    tmp36 = tl.load(in_ptr1 + (96 + r0), None)
    tmp42 = tl.load(in_ptr0 + (128 + r0), None)
    tmp43 = tl.load(in_ptr1 + (128 + r0), None)
    tmp49 = tl.load(in_ptr0 + (144 + r0), None)
    tmp50 = tl.load(in_ptr1 + (144 + r0), None)
    tmp56 = tl.load(in_ptr0 + (160 + r0), None)
    tmp57 = tl.load(in_ptr1 + (160 + r0), None)
    tmp63 = tl.load(in_ptr0 + (192 + r0), None)
    tmp64 = tl.load(in_ptr1 + (192 + r0), None)
    tmp70 = tl.load(in_ptr0 + (208 + r0), None)
    tmp71 = tl.load(in_ptr1 + (208 + r0), None)
    tmp77 = tl.load(in_ptr0 + (224 + r0), None)
    tmp78 = tl.load(in_ptr1 + (224 + r0), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None]
    tmp9 = tmp7 - tmp8
    tmp10 = tl_math.abs(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None]
    tmp16 = tmp14 - tmp15
    tmp17 = tl_math.abs(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tmp23 = tmp21 - tmp22
    tmp24 = tl_math.abs(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp30 = tmp28 - tmp29
    tmp31 = tl_math.abs(tmp30)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.sum(tmp32, 1)[:, None]
    tmp37 = tmp35 - tmp36
    tmp38 = tl_math.abs(tmp37)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.sum(tmp39, 1)[:, None]
    tmp44 = tmp42 - tmp43
    tmp45 = tl_math.abs(tmp44)
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp48 = tl.sum(tmp46, 1)[:, None]
    tmp51 = tmp49 - tmp50
    tmp52 = tl_math.abs(tmp51)
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
    tmp55 = tl.sum(tmp53, 1)[:, None]
    tmp58 = tmp56 - tmp57
    tmp59 = tl_math.abs(tmp58)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, RBLOCK])
    tmp62 = tl.sum(tmp60, 1)[:, None]
    tmp65 = tmp63 - tmp64
    tmp66 = tl_math.abs(tmp65)
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
    tmp69 = tl.sum(tmp67, 1)[:, None]
    tmp72 = tmp70 - tmp71
    tmp73 = tl_math.abs(tmp72)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, RBLOCK])
    tmp76 = tl.sum(tmp74, 1)[:, None]
    tmp79 = tmp77 - tmp78
    tmp80 = tl_math.abs(tmp79)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp83 = tl.sum(tmp81, 1)[:, None]
    tmp84 = 16.0
    tmp85 = tmp6 / tmp84
    tmp86 = 0.5
    tmp87 = tmp85 * tmp86
    tmp88 = 4.0
    tmp89 = tmp87 * tmp88
    tmp90 = 0.0
    tmp91 = tmp89 + tmp90
    tmp92 = tmp13 / tmp84
    tmp93 = tmp92 * tmp86
    tmp94 = tmp93 * tmp88
    tmp95 = tmp91 + tmp94
    tmp96 = tmp20 / tmp84
    tmp97 = tmp96 * tmp86
    tmp98 = tmp97 * tmp88
    tmp99 = tmp95 + tmp98
    tmp100 = tmp27 / tmp84
    tmp101 = tmp100 * tmp86
    tmp102 = tmp101 * tmp88
    tmp103 = tmp99 + tmp102
    tmp104 = tmp34 / tmp84
    tmp105 = tmp104 * tmp86
    tmp106 = tmp105 * tmp88
    tmp107 = tmp103 + tmp106
    tmp108 = tmp41 / tmp84
    tmp109 = tmp108 * tmp86
    tmp110 = tmp109 * tmp88
    tmp111 = tmp107 + tmp110
    tmp112 = tmp48 / tmp84
    tmp113 = tmp112 * tmp86
    tmp114 = tmp113 * tmp88
    tmp115 = tmp111 + tmp114
    tmp116 = tmp55 / tmp84
    tmp117 = tmp116 * tmp86
    tmp118 = tmp117 * tmp88
    tmp119 = tmp115 + tmp118
    tmp120 = tmp62 / tmp84
    tmp121 = tmp120 * tmp86
    tmp122 = tmp121 * tmp88
    tmp123 = tmp119 + tmp122
    tmp124 = tmp69 / tmp84
    tmp125 = tmp124 * tmp86
    tmp126 = tmp125 * tmp88
    tmp127 = tmp123 + tmp126
    tmp128 = tmp76 / tmp84
    tmp129 = tmp128 * tmp86
    tmp130 = tmp129 * tmp88
    tmp131 = tmp127 + tmp130
    tmp132 = tmp83 / tmp84
    tmp133 = tmp132 * tmp86
    tmp134 = tmp133 * tmp88
    tmp135 = tmp131 + tmp134
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp135, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf12 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [l1_loss, mul, mul_1, loss_G_GAN_Feat, l1_loss_1, mul_2, mul_3, loss_G_GAN_Feat_1, l1_loss_2, mul_4, mul_5, loss_G_GAN_Feat_2, l1_loss_3, mul_6, mul_7, loss_G_GAN_Feat_3, l1_loss_4, mul_8, mul_9, loss_G_GAN_Feat_4, l1_loss_5, mul_10, mul_11, loss_G_GAN_Feat_5, l1_loss_6, mul_12, mul_13, loss_G_GAN_Feat_6, l1_loss_7, mul_14, mul_15, loss_G_GAN_Feat_7, l1_loss_8, mul_16, mul_17, loss_G_GAN_Feat_8, l1_loss_9, mul_18, mul_19, loss_G_GAN_Feat_9, l1_loss_10, mul_20, mul_21, loss_G_GAN_Feat_10, l1_loss_11, mul_22, mul_23, loss_G_GAN_Feat_11], Original ATen: [aten.sub, aten.abs, aten.mean, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_abs_add_mean_mul_sub_0.run(buf12, arg0_1, arg1_1, 1, 16, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
