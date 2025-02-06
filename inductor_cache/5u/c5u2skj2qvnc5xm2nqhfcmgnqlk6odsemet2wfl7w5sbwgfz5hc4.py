# AOT ID: ['26_forward']
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


# kernel path: inductor_cache/ee/ceedukgvimuqcuy5o5zpildeijabqlvnfpbbq6bfhcqswf7dh2hu.py
# Topologically Sorted Source Nodes: [truediv, relu, mul, x], Original ATen: [aten.div, aten.relu, aten.mul, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   mul => mul
#   relu => relu
#   truediv => div
#   x => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, 1.0), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%div,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, 1.7139588594436646), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mul, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_div_mul_reflection_pad2d_relu_0 = async_compile.triton('triton_poi_fused_div_mul_reflection_pad2d_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_reflection_pad2d_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mul_reflection_pad2d_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = ((xindex // 6) % 6)
    x2 = xindex // 36
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 1.7139588594436646
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


cpp_fused_lift_fresh_prod_1 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(2);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(3);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = static_cast<int64_t>(4);
                        auto tmp9 = tmp3 ? tmp8 : tmp7;
                        tmp_acc0 = tmp_acc0 * tmp9;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


# kernel path: inductor_cache/cl/ccle6j4rmjfgpm3v7a6kxcwbtbo256j5y7ywuuxmhhiwutgotnby.py
# Topologically Sorted Source Nodes: [var_mean, mul_1, tensor, max_1, rsqrt, scale, shift, mul_4, sub], Original ATen: [aten.var_mean, aten.mul, aten.lift_fresh, aten.maximum, aten.rsqrt, aten.sub]
# Source node to ATen node mapping:
#   max_1 => maximum
#   mul_1 => mul_1
#   mul_4 => mul_4
#   rsqrt => rsqrt
#   scale => mul_2
#   shift => mul_3
#   sub => sub_4
#   tensor => full_default
#   var_mean => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_2, [1, 2, 3]), kwargs = {correction: 1, keepdim: True})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem, %prod), kwargs = {})
#   %full_default : [num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 9.999999747378752e-05), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%mul_1, %full_default), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%maximum,), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%rsqrt, %primals_3), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_1, %mul_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %mul_2), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_4, %mul_3), kwargs = {})
triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_2 = async_compile.triton('triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': 'i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 36*x0), rmask & xmask, other=0.0)
    tmp19 = in_ptr1
    tmp25 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 36, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 35.0
    tmp18 = tmp16 / tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = 9.999999747378752e-05
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = libdevice.rsqrt(tmp23)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp0 * tmp26
    tmp28 = tmp10 * tmp26
    tmp29 = tmp27 - tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + 36*x0), tmp29, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zm/czmfppxhrh6r25genoezot5otdr6dv5nn2d3qmmrekomc3zkul7t.py
# Topologically Sorted Source Nodes: [signal], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   signal => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %sub_4, %primals_4, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


cpp_fused_lift_fresh_prod_4 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(2);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(3);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = static_cast<int64_t>(4);
                        auto tmp9 = tmp3 ? tmp8 : tmp7;
                        tmp_acc0 = tmp_acc0 * tmp9;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


# kernel path: inductor_cache/fk/cfkidn5we3i6gom5z7ij4p2g2vwbqqrtzbykppnp4oh4zgqfnffa.py
# Topologically Sorted Source Nodes: [tensor, var_mean_1, mul_5, max_2, rsqrt_1, scale_1, shift_1, mul_8, sub_1], Original ATen: [aten.lift_fresh, aten.var_mean, aten.mul, aten.maximum, aten.rsqrt, aten.sub, aten.eq, aten.lt]
# Source node to ATen node mapping:
#   max_2 => maximum_1
#   mul_5 => mul_5
#   mul_8 => mul_8
#   rsqrt_1 => rsqrt_1
#   scale_1 => mul_6
#   shift_1 => mul_7
#   sub_1 => sub_5
#   tensor => full_default
#   var_mean_1 => var_mean_1
# Graph fragment:
#   %full_default : [num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 9.999999747378752e-05), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_5, [1, 2, 3]), kwargs = {correction: 1, keepdim: True})
#   %mul_5 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_2, %prod_1), kwargs = {})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%mul_5, %full_default), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%maximum_1,), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%rsqrt_1, %primals_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_3, %mul_6), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %mul_6), kwargs = {})
#   %sub_5 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_8, %mul_7), kwargs = {})
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%mul_5, %full_default), kwargs = {})
#   %lt_2 : [num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%mul_5, %full_default), kwargs = {})
triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_5 = async_compile.triton('triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*i1', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 36*x0), rmask & xmask, other=0.0)
    tmp19 = in_ptr1
    tmp27 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 36, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 35.0
    tmp18 = tmp16 / tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = 9.999999747378752e-05
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp21 == tmp22
    tmp26 = tmp21 < tmp22
    tmp28 = tmp24 * tmp27
    tmp29 = tmp0 * tmp28
    tmp30 = tmp10 * tmp28
    tmp31 = tmp29 - tmp30
    tl.store(out_ptr2 + (x0), tmp24, xmask)
    tl.store(out_ptr3 + (x0), tmp25, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr5 + (r1 + 36*x0), tmp31, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z2/cz2ckjuzzav6bz6slaniqpdrcdk56xrton5d6wewfqiwjvs5qopq.py
# Topologically Sorted Source Nodes: [gate, mul_9, x_1, relu_1, mul_11, x_2], Original ATen: [aten.sigmoid, aten.mul, aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   gate => sigmoid
#   mul_11 => mul_11
#   mul_9 => mul_9
#   relu_1 => relu_1
#   x_1 => mul_10
#   x_2 => _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_1,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %sigmoid), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 1.8), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mul_10,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_1, 1.7139588594436646), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mul_11, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_mul_reflection_pad2d_relu_sigmoid_6 = async_compile.triton('triton_poi_fused_mul_reflection_pad2d_relu_sigmoid_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_reflection_pad2d_relu_sigmoid_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_reflection_pad2d_relu_sigmoid_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = ((xindex // 6) % 6)
    x2 = xindex // 36
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 1.8
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = 1.7139588594436646
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


cpp_fused_lift_fresh_prod_7 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(2);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(3);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = static_cast<int64_t>(4);
                        auto tmp9 = tmp3 ? tmp8 : tmp7;
                        tmp_acc0 = tmp_acc0 * tmp9;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


cpp_fused_lift_fresh_prod_8 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(2);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(3);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = static_cast<int64_t>(4);
                        auto tmp9 = tmp3 ? tmp8 : tmp7;
                        tmp_acc0 = tmp_acc0 * tmp9;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


# kernel path: inductor_cache/3s/c3s3qgwpsoc6z7sfum3bzbmbgrdtuzt2xwrcxhwaivy4as5tnugd.py
# Topologically Sorted Source Nodes: [signal_1, conv2d_3, gate_1, mul_20, x_3, x_4, add], Original ATen: [aten.convolution, aten.sigmoid, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add => add
#   conv2d_3 => convolution_3
#   gate_1 => sigmoid_1
#   mul_20 => mul_20
#   signal_1 => convolution_2
#   x_3 => mul_21
#   x_4 => mul_22
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_3, %sub_10, %primals_10, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_3, %sub_11, %primals_13, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_3,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, %sigmoid_1), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, 1.8), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, 0.2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %primals_1), kwargs = {})
triton_poi_fused_add_convolution_mul_sigmoid_9 = async_compile.triton('triton_poi_fused_add_convolution_mul_sigmoid_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sigmoid_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_sigmoid_9(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 1.8
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (x3), tmp5, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_3, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_6, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_9, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_12, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv, relu, mul, x], Original ATen: [aten.div, aten.relu, aten.mul, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_mul_reflection_pad2d_relu_0.run(primals_1, buf0, 576, grid=grid(576), stream=stream0)
    buf1 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_1(buf1)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf5 = reinterpret_tensor(buf3, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf3  # reuse
        buf6 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean, mul_1, tensor, max_1, rsqrt, scale, shift, mul_4, sub], Original ATen: [aten.var_mean, aten.mul, aten.lift_fresh, aten.maximum, aten.rsqrt, aten.sub]
        stream0 = get_raw_stream(0)
        triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_2.run(buf5, primals_2, buf1.item(), primals_3, buf2, buf6, 4, 36, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [signal], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf0, buf6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 4, 4, 4), (64, 16, 4, 1))
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [signal], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf8, primals_4, 256, grid=grid(256), stream=stream0)
        del primals_4
    buf9 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_4(buf9)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf10 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf13 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf39 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.bool)
        buf40 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.bool)
        buf14 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor, var_mean_1, mul_5, max_2, rsqrt_1, scale_1, shift_1, mul_8, sub_1], Original ATen: [aten.lift_fresh, aten.var_mean, aten.mul, aten.maximum, aten.rsqrt, aten.sub, aten.eq, aten.lt]
        stream0 = get_raw_stream(0)
        triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_5.run(primals_5, buf9.item(), primals_6, buf10, buf13, buf39, buf40, buf14, 4, 36, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf0, buf14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 4, 4, 4), (64, 16, 4, 1))
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf16, primals_7, 256, grid=grid(256), stream=stream0)
        del primals_7
        buf17 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [gate, mul_9, x_1, relu_1, mul_11, x_2], Original ATen: [aten.sigmoid, aten.mul, aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reflection_pad2d_relu_sigmoid_6.run(buf8, buf16, buf17, 576, grid=grid(576), stream=stream0)
    buf18 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_7(buf18)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf19 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf22 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf37 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.bool)
        buf38 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.bool)
        buf23 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor, var_mean_2, mul_12, max_3, rsqrt_2, scale_2, shift_2, mul_15, sub_2], Original ATen: [aten.lift_fresh, aten.var_mean, aten.mul, aten.maximum, aten.rsqrt, aten.sub, aten.eq, aten.lt]
        stream0 = get_raw_stream(0)
        triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_5.run(primals_8, buf18.item(), primals_9, buf19, buf22, buf37, buf38, buf23, 4, 36, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [signal_1], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf17, buf23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 4, 4, 4), (64, 16, 4, 1))
    buf26 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_8(buf26)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf27 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf30 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf35 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.bool)
        buf36 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.bool)
        buf31 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor, var_mean_3, mul_16, max_4, rsqrt_3, scale_3, shift_3, mul_19, sub_3], Original ATen: [aten.lift_fresh, aten.var_mean, aten.mul, aten.maximum, aten.rsqrt, aten.sub, aten.eq, aten.lt]
        stream0 = get_raw_stream(0)
        triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_5.run(primals_11, buf26.item(), primals_12, buf27, buf30, buf35, buf36, buf31, 4, 36, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf17, buf31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 4, 4, 4), (64, 16, 4, 1))
        buf25 = buf24; del buf24  # reuse
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [signal_1, conv2d_3, gate_1, mul_20, x_3, x_4, add], Original ATen: [aten.convolution, aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_sigmoid_9.run(buf25, buf33, primals_10, primals_13, primals_1, buf34, 256, grid=grid(256), stream=stream0)
        del primals_1
        del primals_10
        del primals_13
    return (buf34, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, buf0, buf1, buf2, buf5, buf6, buf8, buf9, buf10, buf13, buf14, buf16, buf17, buf18, buf19, buf22, buf23, buf25, buf26, buf27, buf30, buf31, buf33, buf35, buf36, buf37, buf38, buf39, buf40, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
