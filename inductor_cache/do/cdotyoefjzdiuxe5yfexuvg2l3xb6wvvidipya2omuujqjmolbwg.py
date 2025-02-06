# AOT ID: ['5_inference']
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


# kernel path: inductor_cache/72/c72e3vzatuxddci26cyf3udcwdlm6pjjcvarwhawbi4iynas6wtf.py
# Topologically Sorted Source Nodes: [add, sub, scale1, reciprocal, mul, truediv, erf, add_1, mul_1, sub_1, sub_2, reciprocal_1, mul_2, truediv_1, erf_1, add_2, mul_3, sub_3, likelihood1, mul_16, add_3, sub_4, scale2, reciprocal_2, mul_4, truediv_2, erf_2, add_4, mul_5, sub_5, sub_6, reciprocal_3, mul_6, truediv_3, erf_3, add_5, mul_7, sub_7, likelihood2, mul_17, add_12, add_6, sub_8, scale3, reciprocal_4, mul_8, truediv_4, erf_4, add_7, mul_9, sub_9, sub_10, reciprocal_5, mul_10, truediv_5, erf_5, add_8, mul_11, sub_11, likelihood3, mul_18, add_13, add_9, sub_12, scale4, reciprocal_6, mul_12, truediv_6, erf_6, add_10, mul_13, sub_13, sub_14, reciprocal_7, mul_14, truediv_7, erf_7, add_11, mul_15, sub_15, likelihood4, mul_19, add_14, x, x_1, log2, bits], Original ATen: [aten.add, aten.sub, aten.clamp, aten.reciprocal, aten.mul, aten.div, aten.erf, aten.abs, aten.log2, aten.neg]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_10 => add_10
#   add_11 => add_11
#   add_12 => add_12
#   add_13 => add_13
#   add_14 => add_14
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   add_7 => add_7
#   add_8 => add_8
#   add_9 => add_9
#   bits => neg
#   erf => erf
#   erf_1 => erf_1
#   erf_2 => erf_2
#   erf_3 => erf_3
#   erf_4 => erf_4
#   erf_5 => erf_5
#   erf_6 => erf_6
#   erf_7 => erf_7
#   likelihood1 => abs_1
#   likelihood2 => abs_2
#   likelihood3 => abs_3
#   likelihood4 => abs_4
#   log2 => log2
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
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   reciprocal => reciprocal
#   reciprocal_1 => reciprocal_1
#   reciprocal_2 => reciprocal_2
#   reciprocal_3 => reciprocal_3
#   reciprocal_4 => reciprocal_4
#   reciprocal_5 => reciprocal_5
#   reciprocal_6 => reciprocal_6
#   reciprocal_7 => reciprocal_7
#   scale1 => clamp_min
#   scale2 => clamp_min_1
#   scale3 => clamp_min_2
#   scale4 => clamp_min_3
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sub_15 => sub_15
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sub_9 => sub_9
#   truediv => div
#   truediv_1 => div_1
#   truediv_2 => div_2
#   truediv_3 => div_3
#   truediv_4 => div_4
#   truediv_5 => div_5
#   truediv_6 => div_6
#   truediv_7 => div_7
#   x => clamp_min_4
#   x_1 => clamp_min_5
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 0.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %arg4_1), kwargs = {})
#   %clamp_min : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg0_1, 1e-09), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %reciprocal), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 1.4142135623730951), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%div,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg8_1, 0.5), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %arg4_1), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %reciprocal_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, 1.4142135623730951), kwargs = {})
#   %erf_1 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%div_1,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.5), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, %mul_3), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_3,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg9_1, %abs_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 0.5), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %arg5_1), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg1_1, 1e-09), kwargs = {})
#   %reciprocal_2 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_1,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %reciprocal_2), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_4, 1.4142135623730951), kwargs = {})
#   %erf_2 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%div_2,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.5), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg8_1, 0.5), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_5, %arg5_1), kwargs = {})
#   %reciprocal_3 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_1,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %reciprocal_3), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, 1.4142135623730951), kwargs = {})
#   %erf_3 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%div_3,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.5), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_5, %mul_7), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_7,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg10_1, %abs_2), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %mul_17), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 0.5), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %arg6_1), kwargs = {})
#   %clamp_min_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg2_1, 1e-09), kwargs = {})
#   %reciprocal_4 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_2,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %reciprocal_4), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_8, 1.4142135623730951), kwargs = {})
#   %erf_4 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%div_4,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.5), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg8_1, 0.5), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %arg6_1), kwargs = {})
#   %reciprocal_5 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_2,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %reciprocal_5), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_10, 1.4142135623730951), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%div_5,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 0.5), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %mul_11), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_11,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg11_1, %abs_3), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mul_18), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 0.5), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %arg7_1), kwargs = {})
#   %clamp_min_3 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg3_1, 1e-09), kwargs = {})
#   %reciprocal_6 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_3,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %reciprocal_6), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_12, 1.4142135623730951), kwargs = {})
#   %erf_6 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%div_6,), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg8_1, 0.5), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_13, %arg7_1), kwargs = {})
#   %reciprocal_7 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_3,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %reciprocal_7), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_14, 1.4142135623730951), kwargs = {})
#   %erf_7 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%div_7,), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_7, 1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, 0.5), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_13, %mul_15), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_15,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg12_1, %abs_4), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %mul_19), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_14, 1e-06), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%clamp_min_4, 1e-06), kwargs = {})
#   %log2 : [num_users=1] = call_function[target=torch.ops.aten.log2.default](args = (%clamp_min_5,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%log2,), kwargs = {})
triton_poi_fused_abs_add_clamp_div_erf_log2_mul_neg_reciprocal_sub_0 = async_compile.triton('triton_poi_fused_abs_add_clamp_div_erf_log2_mul_neg_reciprocal_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_clamp_div_erf_log2_mul_neg_reciprocal_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_clamp_div_erf_log2_mul_neg_reciprocal_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    tmp28 = tl.load(in_ptr4 + (x0), xmask)
    tmp29 = tl.load(in_ptr5 + (x0), xmask)
    tmp31 = tl.load(in_ptr6 + (x0), xmask)
    tmp49 = tl.load(in_ptr7 + (x0), xmask)
    tmp50 = tl.load(in_ptr8 + (x0), xmask)
    tmp52 = tl.load(in_ptr9 + (x0), xmask)
    tmp70 = tl.load(in_ptr10 + (x0), xmask)
    tmp71 = tl.load(in_ptr11 + (x0), xmask)
    tmp73 = tl.load(in_ptr12 + (x0), xmask)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 - tmp4
    tmp7 = 1e-09
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp5 * tmp10
    tmp12 = 0.7071067811865475
    tmp13 = tmp11 * tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 * tmp2
    tmp18 = tmp1 - tmp2
    tmp19 = tmp18 - tmp4
    tmp20 = tmp19 * tmp10
    tmp21 = tmp20 * tmp12
    tmp22 = libdevice.erf(tmp21)
    tmp23 = tmp22 + tmp15
    tmp24 = tmp23 * tmp2
    tmp25 = tmp17 - tmp24
    tmp26 = tl_math.abs(tmp25)
    tmp27 = tmp0 * tmp26
    tmp30 = tmp3 - tmp29
    tmp32 = triton_helpers.maximum(tmp31, tmp7)
    tmp33 = tmp9 / tmp32
    tmp34 = tmp30 * tmp33
    tmp35 = tmp34 * tmp12
    tmp36 = libdevice.erf(tmp35)
    tmp37 = tmp36 + tmp15
    tmp38 = tmp37 * tmp2
    tmp39 = tmp18 - tmp29
    tmp40 = tmp39 * tmp33
    tmp41 = tmp40 * tmp12
    tmp42 = libdevice.erf(tmp41)
    tmp43 = tmp42 + tmp15
    tmp44 = tmp43 * tmp2
    tmp45 = tmp38 - tmp44
    tmp46 = tl_math.abs(tmp45)
    tmp47 = tmp28 * tmp46
    tmp48 = tmp27 + tmp47
    tmp51 = tmp3 - tmp50
    tmp53 = triton_helpers.maximum(tmp52, tmp7)
    tmp54 = tmp9 / tmp53
    tmp55 = tmp51 * tmp54
    tmp56 = tmp55 * tmp12
    tmp57 = libdevice.erf(tmp56)
    tmp58 = tmp57 + tmp15
    tmp59 = tmp58 * tmp2
    tmp60 = tmp18 - tmp50
    tmp61 = tmp60 * tmp54
    tmp62 = tmp61 * tmp12
    tmp63 = libdevice.erf(tmp62)
    tmp64 = tmp63 + tmp15
    tmp65 = tmp64 * tmp2
    tmp66 = tmp59 - tmp65
    tmp67 = tl_math.abs(tmp66)
    tmp68 = tmp49 * tmp67
    tmp69 = tmp48 + tmp68
    tmp72 = tmp3 - tmp71
    tmp74 = triton_helpers.maximum(tmp73, tmp7)
    tmp75 = tmp9 / tmp74
    tmp76 = tmp72 * tmp75
    tmp77 = tmp76 * tmp12
    tmp78 = libdevice.erf(tmp77)
    tmp79 = tmp78 + tmp15
    tmp80 = tmp79 * tmp2
    tmp81 = tmp18 - tmp71
    tmp82 = tmp81 * tmp75
    tmp83 = tmp82 * tmp12
    tmp84 = libdevice.erf(tmp83)
    tmp85 = tmp84 + tmp15
    tmp86 = tmp85 * tmp2
    tmp87 = tmp80 - tmp86
    tmp88 = tl_math.abs(tmp87)
    tmp89 = tmp70 * tmp88
    tmp90 = tmp69 + tmp89
    tmp91 = 1e-06
    tmp92 = triton_helpers.maximum(tmp90, tmp91)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = libdevice.log2(tmp93)
    tmp95 = -tmp94
    tl.store(in_out_ptr0 + (x0), tmp95, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg3_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg4_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg5_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg6_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg7_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg8_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg9_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg10_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg11_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg12_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [add, sub, scale1, reciprocal, mul, truediv, erf, add_1, mul_1, sub_1, sub_2, reciprocal_1, mul_2, truediv_1, erf_1, add_2, mul_3, sub_3, likelihood1, mul_16, add_3, sub_4, scale2, reciprocal_2, mul_4, truediv_2, erf_2, add_4, mul_5, sub_5, sub_6, reciprocal_3, mul_6, truediv_3, erf_3, add_5, mul_7, sub_7, likelihood2, mul_17, add_12, add_6, sub_8, scale3, reciprocal_4, mul_8, truediv_4, erf_4, add_7, mul_9, sub_9, sub_10, reciprocal_5, mul_10, truediv_5, erf_5, add_8, mul_11, sub_11, likelihood3, mul_18, add_13, add_9, sub_12, scale4, reciprocal_6, mul_12, truediv_6, erf_6, add_10, mul_13, sub_13, sub_14, reciprocal_7, mul_14, truediv_7, erf_7, add_11, mul_15, sub_15, likelihood4, mul_19, add_14, x, x_1, log2, bits], Original ATen: [aten.add, aten.sub, aten.clamp, aten.reciprocal, aten.mul, aten.div, aten.erf, aten.abs, aten.log2, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_clamp_div_erf_log2_mul_neg_reciprocal_sub_0.run(buf2, arg9_1, arg8_1, arg4_1, arg0_1, arg10_1, arg5_1, arg1_1, arg11_1, arg6_1, arg2_1, arg12_1, arg7_1, arg3_1, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg10_1
        del arg11_1
        del arg12_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
