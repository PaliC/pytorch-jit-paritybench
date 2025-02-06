# AOT ID: ['0_inference']
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


# kernel path: inductor_cache/ch/cchn25i7zfh35rv55ajmpzv4jfvltnwxpuwfbfva4x4dmsdlc2os.py
# Topologically Sorted Source Nodes: [neg_dist_1, max_1, neg_dist_2], Original ATen: [aten.div, aten.max, aten.sub]
# Source node to ATen node mapping:
#   max_1 => max_1
#   neg_dist_1 => div_1
#   neg_dist_2 => sub_1
# Graph fragment:
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_7, 1.0), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%div_1, 1, True), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %getitem), kwargs = {})
triton_poi_fused_div_max_sub_0 = async_compile.triton('triton_poi_fused_div_max_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_max_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_max_sub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/37/c37skc5chycnmh3rycz7vrqo7lgfr6qrxsivx4md3sg3tu3rdwbp.py
# Topologically Sorted Source Nodes: [neg_dist_1, max_1, pos_dist_1, pos_dist_2, neg, align, logsumexp, uniform, add_1, c_mean, align_corrected, uniform_corrected], Original ATen: [aten.div, aten.max, aten.sub, aten.neg, aten.mean, aten.logsumexp, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_2
#   align => mean
#   align_corrected => sub_3
#   c_mean => mean_2
#   logsumexp => abs_1, add, amax, eq, exp, full_default, log, sub_2, sum_1, where
#   max_1 => max_1
#   neg => neg
#   neg_dist_1 => div_1
#   pos_dist_1 => div
#   pos_dist_2 => sub
#   uniform => mean_1
#   uniform_corrected => add_1
# Graph fragment:
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_7, 1.0), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%div_1, 1, True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_3, 1.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %squeeze), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sub,), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%neg,), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%sub_1, [1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %where), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze_1), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%add,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mean_1), kwargs = {})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%getitem,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %mean_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, %mean_2), kwargs = {})
triton_poi_fused_add_div_logsumexp_max_mean_neg_sub_1 = async_compile.triton('triton_poi_fused_add_div_logsumexp_max_mean_neg_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_logsumexp_max_mean_neg_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 36, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_logsumexp_max_mean_neg_sub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (2))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (1))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (4))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp27 = tl.load(in_ptr1 + (5))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp31 = tl.load(in_ptr1 + (6))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp35 = tl.load(in_ptr1 + (7))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp42 = tl.load(in_ptr0 + (2))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp45 = tl.load(in_ptr1 + (8))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp48 = tl.load(in_ptr1 + (9))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
    tmp52 = tl.load(in_ptr1 + (10))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp56 = tl.load(in_ptr1 + (11))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp63 = tl.load(in_ptr0 + (3))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp66 = tl.load(in_ptr1 + (12))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp69 = tl.load(in_ptr1 + (13))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp73 = tl.load(in_ptr1 + (14))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp77 = tl.load(in_ptr1 + (15))
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK])
    tmp91 = tl.load(in_ptr2 + (0))
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK])
    tmp93 = tl.load(in_ptr2 + (1))
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK])
    tmp96 = tl.load(in_ptr2 + (2))
    tmp97 = tl.broadcast_to(tmp96, [XBLOCK])
    tmp99 = tl.load(in_ptr2 + (3))
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK])
    tmp120 = tl.load(in_ptr2 + (4))
    tmp121 = tl.broadcast_to(tmp120, [XBLOCK])
    tmp122 = tl.load(in_ptr2 + (5))
    tmp123 = tl.broadcast_to(tmp122, [XBLOCK])
    tmp125 = tl.load(in_ptr2 + (6))
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK])
    tmp128 = tl.load(in_ptr2 + (7))
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK])
    tmp148 = tl.load(in_ptr2 + (8))
    tmp149 = tl.broadcast_to(tmp148, [XBLOCK])
    tmp150 = tl.load(in_ptr2 + (9))
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK])
    tmp153 = tl.load(in_ptr2 + (10))
    tmp154 = tl.broadcast_to(tmp153, [XBLOCK])
    tmp156 = tl.load(in_ptr2 + (11))
    tmp157 = tl.broadcast_to(tmp156, [XBLOCK])
    tmp176 = tl.load(in_ptr2 + (12))
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK])
    tmp178 = tl.load(in_ptr2 + (13))
    tmp179 = tl.broadcast_to(tmp178, [XBLOCK])
    tmp181 = tl.load(in_ptr2 + (14))
    tmp182 = tl.broadcast_to(tmp181, [XBLOCK])
    tmp184 = tl.load(in_ptr2 + (15))
    tmp185 = tl.broadcast_to(tmp184, [XBLOCK])
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8 * tmp2
    tmp10 = triton_helpers.maximum(tmp6, tmp9)
    tmp13 = tmp12 * tmp2
    tmp14 = triton_helpers.maximum(tmp10, tmp13)
    tmp17 = tmp16 * tmp2
    tmp18 = triton_helpers.maximum(tmp14, tmp17)
    tmp19 = tmp3 - tmp18
    tmp20 = -tmp19
    tmp23 = tmp22 * tmp2
    tmp26 = tmp25 * tmp2
    tmp29 = tmp28 * tmp2
    tmp30 = triton_helpers.maximum(tmp26, tmp29)
    tmp33 = tmp32 * tmp2
    tmp34 = triton_helpers.maximum(tmp30, tmp33)
    tmp37 = tmp36 * tmp2
    tmp38 = triton_helpers.maximum(tmp34, tmp37)
    tmp39 = tmp23 - tmp38
    tmp40 = -tmp39
    tmp41 = tmp20 + tmp40
    tmp44 = tmp43 * tmp2
    tmp47 = tmp46 * tmp2
    tmp50 = tmp49 * tmp2
    tmp51 = triton_helpers.maximum(tmp47, tmp50)
    tmp54 = tmp53 * tmp2
    tmp55 = triton_helpers.maximum(tmp51, tmp54)
    tmp58 = tmp57 * tmp2
    tmp59 = triton_helpers.maximum(tmp55, tmp58)
    tmp60 = tmp44 - tmp59
    tmp61 = -tmp60
    tmp62 = tmp41 + tmp61
    tmp65 = tmp64 * tmp2
    tmp68 = tmp67 * tmp2
    tmp71 = tmp70 * tmp2
    tmp72 = triton_helpers.maximum(tmp68, tmp71)
    tmp75 = tmp74 * tmp2
    tmp76 = triton_helpers.maximum(tmp72, tmp75)
    tmp79 = tmp78 * tmp2
    tmp80 = triton_helpers.maximum(tmp76, tmp79)
    tmp81 = tmp65 - tmp80
    tmp82 = -tmp81
    tmp83 = tmp62 + tmp82
    tmp84 = 4.0
    tmp85 = tmp83 / tmp84
    tmp86 = tmp18 + tmp38
    tmp87 = tmp86 + tmp59
    tmp88 = tmp87 + tmp80
    tmp89 = tmp88 / tmp84
    tmp90 = tmp85 - tmp89
    tmp95 = triton_helpers.maximum(tmp92, tmp94)
    tmp98 = triton_helpers.maximum(tmp95, tmp97)
    tmp101 = triton_helpers.maximum(tmp98, tmp100)
    tmp102 = tl_math.abs(tmp101)
    tmp103 = float("inf")
    tmp104 = tmp102 == tmp103
    tmp105 = 0.0
    tmp106 = tl.where(tmp104, tmp105, tmp101)
    tmp107 = tmp92 - tmp106
    tmp108 = tl_math.exp(tmp107)
    tmp109 = tmp94 - tmp106
    tmp110 = tl_math.exp(tmp109)
    tmp111 = tmp108 + tmp110
    tmp112 = tmp97 - tmp106
    tmp113 = tl_math.exp(tmp112)
    tmp114 = tmp111 + tmp113
    tmp115 = tmp100 - tmp106
    tmp116 = tl_math.exp(tmp115)
    tmp117 = tmp114 + tmp116
    tmp118 = tl_math.log(tmp117)
    tmp119 = tmp118 + tmp106
    tmp124 = triton_helpers.maximum(tmp121, tmp123)
    tmp127 = triton_helpers.maximum(tmp124, tmp126)
    tmp130 = triton_helpers.maximum(tmp127, tmp129)
    tmp131 = tl_math.abs(tmp130)
    tmp132 = tmp131 == tmp103
    tmp133 = tl.where(tmp132, tmp105, tmp130)
    tmp134 = tmp121 - tmp133
    tmp135 = tl_math.exp(tmp134)
    tmp136 = tmp123 - tmp133
    tmp137 = tl_math.exp(tmp136)
    tmp138 = tmp135 + tmp137
    tmp139 = tmp126 - tmp133
    tmp140 = tl_math.exp(tmp139)
    tmp141 = tmp138 + tmp140
    tmp142 = tmp129 - tmp133
    tmp143 = tl_math.exp(tmp142)
    tmp144 = tmp141 + tmp143
    tmp145 = tl_math.log(tmp144)
    tmp146 = tmp145 + tmp133
    tmp147 = tmp119 + tmp146
    tmp152 = triton_helpers.maximum(tmp149, tmp151)
    tmp155 = triton_helpers.maximum(tmp152, tmp154)
    tmp158 = triton_helpers.maximum(tmp155, tmp157)
    tmp159 = tl_math.abs(tmp158)
    tmp160 = tmp159 == tmp103
    tmp161 = tl.where(tmp160, tmp105, tmp158)
    tmp162 = tmp149 - tmp161
    tmp163 = tl_math.exp(tmp162)
    tmp164 = tmp151 - tmp161
    tmp165 = tl_math.exp(tmp164)
    tmp166 = tmp163 + tmp165
    tmp167 = tmp154 - tmp161
    tmp168 = tl_math.exp(tmp167)
    tmp169 = tmp166 + tmp168
    tmp170 = tmp157 - tmp161
    tmp171 = tl_math.exp(tmp170)
    tmp172 = tmp169 + tmp171
    tmp173 = tl_math.log(tmp172)
    tmp174 = tmp173 + tmp161
    tmp175 = tmp147 + tmp174
    tmp180 = triton_helpers.maximum(tmp177, tmp179)
    tmp183 = triton_helpers.maximum(tmp180, tmp182)
    tmp186 = triton_helpers.maximum(tmp183, tmp185)
    tmp187 = tl_math.abs(tmp186)
    tmp188 = tmp187 == tmp103
    tmp189 = tl.where(tmp188, tmp105, tmp186)
    tmp190 = tmp177 - tmp189
    tmp191 = tl_math.exp(tmp190)
    tmp192 = tmp179 - tmp189
    tmp193 = tl_math.exp(tmp192)
    tmp194 = tmp191 + tmp193
    tmp195 = tmp182 - tmp189
    tmp196 = tl_math.exp(tmp195)
    tmp197 = tmp194 + tmp196
    tmp198 = tmp185 - tmp189
    tmp199 = tl_math.exp(tmp198)
    tmp200 = tmp197 + tmp199
    tmp201 = tl_math.log(tmp200)
    tmp202 = tmp201 + tmp189
    tmp203 = tmp175 + tmp202
    tmp204 = tmp203 / tmp84
    tmp205 = tmp85 + tmp204
    tmp206 = tmp204 + tmp89
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp90, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp205, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp206, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    assert_size_stride(arg2_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [neg_dist], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (1, 4, 4), (16, 4, 1), 0), reinterpret_tensor(arg2_1, (1, 4, 4), (0, 1, 4), 0), out=buf0)
        del arg2_1
        buf1 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pos_dist], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 1, 4), (4, 4, 1), 0), reinterpret_tensor(arg1_1, (4, 4, 1), (4, 1, 1), 0), out=buf1)
        del arg0_1
        del arg1_1
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [neg_dist_1, max_1, neg_dist_2], Original ATen: [aten.div, aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_max_sub_0.run(buf0, buf3, 16, grid=grid(16), stream=stream0)
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf6 = empty_strided_cuda((), (), torch.float32)
        buf8 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [neg_dist_1, max_1, pos_dist_1, pos_dist_2, neg, align, logsumexp, uniform, add_1, c_mean, align_corrected, uniform_corrected], Original ATen: [aten.div, aten.max, aten.sub, aten.neg, aten.mean, aten.logsumexp, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_logsumexp_max_mean_neg_sub_1.run(buf1, buf0, buf3, buf7, buf6, buf8, 1, grid=grid(1), stream=stream0)
        del buf0
        del buf1
        del buf3
    return (buf6, buf7, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
