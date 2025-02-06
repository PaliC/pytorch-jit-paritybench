# AOT ID: ['1_forward']
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


cpp_fused_cat_0 = async_compile.cpp_pybinding(['int64_t*', 'int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0,
                       int64_t* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(9L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = (-1L) + (c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(3L)));
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(9L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = (-1L) + ((static_cast<int64_t>(x0) % static_cast<int64_t>(3L)));
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    out_ptr1[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
}
''')


cpp_fused_repeat_1 = async_compile.cpp_pybinding(['int64_t*', 'int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0,
                       int64_t* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(9L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = 1L + x1;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            out_ptr0[static_cast<int64_t>(x2 + 64L*x1 + 4096L*x0)] = tmp1;
                        }
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(576L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = 1L + x1;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        out_ptr1[static_cast<int64_t>(x1 + 64L*x0)] = tmp1;
                    }
                }
            }
        }
    }
}
''')


# kernel path: inductor_cache/jm/cjmygubvj4dof2bhlti2wikdnizz73ficzuur4zpx4z5cvxyx5ln.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.ge, aten.le, aten.logical_and]
# Source node to ATen node mapping:
# Graph fragment:
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%slice_10, 0), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%slice_10, 65), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %le), kwargs = {})
triton_poi_fused_ge_le_logical_and_2 = async_compile.triton('triton_poi_fused_ge_le_logical_and_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_le_logical_and_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ge_le_logical_and_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = (xindex % 36864)
    x1 = ((xindex // 4096) % 9)
    x2 = xindex // 36864
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (36864 + x3), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (9 + x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (36864 + x3 + 73728*x2), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 0.0
    tmp8 = tmp6 >= tmp7
    tmp9 = 65.0
    tmp10 = tmp6 <= tmp9
    tmp11 = tmp8 & tmp10
    tl.store(out_ptr0 + (x4), tmp11, None)
''', device_str='cuda')


# kernel path: inductor_cache/dj/cdj6y55zxx2irol3l67bm4x7e6lc2l7h64o5iujvp6seqd7f4ydd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.ge, aten.le, aten.logical_and]
# Source node to ATen node mapping:
# Graph fragment:
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%slice_9, 0), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%slice_9, 65), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge_1, %le_1), kwargs = {})
triton_poi_fused_ge_le_logical_and_3 = async_compile.triton('triton_poi_fused_ge_le_logical_and_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_le_logical_and_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ge_le_logical_and_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = (xindex % 36864)
    x1 = ((xindex // 4096) % 9)
    x2 = xindex // 36864
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x3 + 73728*x2), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 0.0
    tmp8 = tmp6 >= tmp7
    tmp9 = 65.0
    tmp10 = tmp6 <= tmp9
    tmp11 = tmp8 & tmp10
    tl.store(out_ptr0 + (x4), tmp11, None)
''', device_str='cuda')


# kernel path: inductor_cache/gg/cggxtlnbiwd4awln3w7qpphbadxqiyptuvdrtaftmbvy4gulaliw.py
# Topologically Sorted Source Nodes: [cat_2, cat_3, p_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
#   cat_3 => cat_3
#   p_2 => cat_6
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max, %clamp_max_1], -1), kwargs = {})
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_2, %clamp_max_3], -1), kwargs = {})
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_4, %clamp_max_5], -1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 18)
    x1 = ((xindex // 18) % 4096)
    x2 = xindex // 73728
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 9, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 4096*(x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.load(in_ptr2 + (x1 + 4096*(x0) + 73728*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.floor(tmp11)
    tmp13 = 0.0
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = 65.0
    tmp16 = triton_helpers.minimum(tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 18, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (36864 + x1 + 4096*((-9) + x0)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (9 + ((-9) + x0)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 + tmp25
    tmp27 = tl.load(in_ptr2 + (36864 + x1 + 4096*((-9) + x0) + 73728*x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.floor(tmp28)
    tmp30 = 0.0
    tmp31 = triton_helpers.maximum(tmp29, tmp30)
    tmp32 = 65.0
    tmp33 = triton_helpers.minimum(tmp31, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp18, tmp35)
    tmp37 = 1.0
    tmp38 = tmp12 + tmp37
    tmp39 = triton_helpers.maximum(tmp38, tmp13)
    tmp40 = triton_helpers.minimum(tmp39, tmp15)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp4, tmp40, tmp41)
    tmp43 = 1.0
    tmp44 = tmp29 + tmp43
    tmp45 = triton_helpers.maximum(tmp44, tmp30)
    tmp46 = triton_helpers.minimum(tmp45, tmp32)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp19, tmp46, tmp47)
    tmp49 = tl.where(tmp4, tmp42, tmp48)
    tmp50 = triton_helpers.maximum(tmp11, tmp13)
    tmp51 = triton_helpers.minimum(tmp50, tmp15)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp4, tmp51, tmp52)
    tmp54 = triton_helpers.maximum(tmp28, tmp30)
    tmp55 = triton_helpers.minimum(tmp54, tmp32)
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp19, tmp55, tmp56)
    tmp58 = tl.where(tmp4, tmp53, tmp57)
    tl.store(out_ptr0 + (x4), tmp36, None)
    tl.store(out_ptr1 + (x4), tmp49, None)
    tl.store(out_ptr2 + (x4), tmp58, None)
''', device_str='cuda')


# kernel path: inductor_cache/ys/cysftudg5got7zbarra4pujgegtq5yb45nzryu3feeot5l5awntg.py
# Topologically Sorted Source Nodes: [type_as, sub, add_3, type_as_1, sub_1, add_4, type_as_2, sub_2, sub_3, type_as_3, sub_4, sub_5, type_as_4, sub_6, add_5, type_as_5, sub_7, sub_8, type_as_6, sub_9, sub_10, type_as_7, sub_11, add_6, gather, gather_1, gather_2, gather_3, mul_8, mul_9, add_11, mul_10, add_12, mul_11, x_offset_4, x_offset_5], Original ATen: [aten._to_copy, aten.sub, aten.add, aten.rsub, aten.gather, aten.mul]
# Source node to ATen node mapping:
#   add_11 => add_11
#   add_12 => add_12
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   gather => gather
#   gather_1 => gather_1
#   gather_2 => gather_2
#   gather_3 => gather_3
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_8 => mul_8
#   mul_9 => mul_9
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sub_9 => sub_9
#   type_as => convert_element_type_4
#   type_as_1 => convert_element_type_5
#   type_as_2 => convert_element_type_6
#   type_as_3 => convert_element_type_7
#   type_as_4 => convert_element_type_8
#   type_as_5 => convert_element_type_9
#   type_as_6 => convert_element_type_10
#   type_as_7 => convert_element_type_11
#   x_offset_4 => add_13
#   x_offset_5 => mul_12
# Graph fragment:
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_5, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %slice_12), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, 1), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_8, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_5, %slice_14), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_1, 1), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_7, torch.float32), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %slice_12), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sub_2), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_6, torch.float32), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_7, %slice_14), kwargs = {})
#   %sub_5 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sub_4), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_19, torch.float32), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_8, %slice_12), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_6, 1), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_21, torch.float32), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_9, %slice_14), kwargs = {})
#   %sub_8 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sub_7), kwargs = {})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_23, torch.float32), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_10, %slice_12), kwargs = {})
#   %sub_10 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sub_9), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_25, torch.float32), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_11, %slice_14), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_11, 1), kwargs = {})
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_11, -1, %view_12), kwargs = {})
#   %gather_1 : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_11, -1, %view_15), kwargs = {})
#   %gather_2 : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_11, -1, %view_18), kwargs = {})
#   %gather_3 : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%view_11, -1, %view_21), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_4, %view_13), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_16), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_6, %view_19), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_22), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mul_11), kwargs = {})
#   %mul_12 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %unsqueeze_8), kwargs = {})
triton_poi_fused__to_copy_add_gather_mul_rsub_sub_5 = async_compile.triton('triton_poi_fused__to_copy_add_gather_mul_rsub_sub_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_gather_mul_rsub_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_gather_mul_rsub_sub_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y3 = yindex // 4096
    y2 = (yindex % 4096)
    tmp0 = tl.load(in_ptr0 + (x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr0 + (9 + x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr1 + (9 + x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr2 + (9 + x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp170 = tl.load(in_ptr4 + (y2 + 4096*x1 + 36864*y3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 - tmp3
    tmp11 = tmp5 - tmp10
    tmp12 = x1
    tmp13 = tl.full([1, 1], 0, tl.int64)
    tmp14 = tmp12 >= tmp13
    tmp15 = tl.full([1, 1], 9, tl.int64)
    tmp16 = tmp12 < tmp15
    tmp17 = tl.load(in_ptr0 + (18*y0 + (x1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17.to(tl.int64)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp16, tmp18, tmp19)
    tmp21 = tmp12 >= tmp15
    tmp22 = tl.full([1, 1], 18, tl.int64)
    tmp23 = tmp12 < tmp22
    tmp24 = tl.load(in_ptr2 + (9 + 18*y0 + ((-9) + x1)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24.to(tl.int64)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp21, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 - tmp3
    tmp31 = tmp30 + tmp5
    tmp32 = tl.load(in_ptr2 + (18*y0 + (x1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32.to(tl.int64)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp16, tmp33, tmp34)
    tmp36 = tl.load(in_ptr0 + (9 + 18*y0 + ((-9) + x1)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp36.to(tl.int64)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp16, tmp35, tmp39)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 - tmp3
    tmp43 = tmp5 - tmp42
    tmp45 = tmp44.to(tl.int64)
    tmp46 = tmp45.to(tl.float32)
    tmp48 = tmp46 - tmp47
    tmp49 = tmp48 + tmp5
    tmp51 = tmp50.to(tl.int64)
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp52 - tmp47
    tmp54 = tmp5 - tmp53
    tmp55 = 9 + x1
    tmp56 = tmp55 >= tmp13
    tmp57 = tmp55 < tmp15
    tmp58 = tl.load(in_ptr0 + (18*y0 + (9 + x1)), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58.to(tl.int64)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp57, tmp59, tmp60)
    tmp62 = tmp55 >= tmp15
    tmp63 = tmp55 < tmp22
    tmp64 = tl.load(in_ptr2 + (9 + 18*y0 + (x1)), tmp62 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp64.to(tl.int64)
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp62, tmp65, tmp66)
    tmp68 = tl.where(tmp57, tmp61, tmp67)
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp69 - tmp47
    tmp71 = tmp5 - tmp70
    tmp72 = tl.load(in_ptr2 + (18*y0 + (9 + x1)), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tmp72.to(tl.int64)
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp57, tmp73, tmp74)
    tmp76 = tl.load(in_ptr0 + (9 + 18*y0 + (x1)), tmp62 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp76.to(tl.int64)
    tmp78 = tl.full(tmp77.shape, 0.0, tmp77.dtype)
    tmp79 = tl.where(tmp62, tmp77, tmp78)
    tmp80 = tl.where(tmp57, tmp75, tmp79)
    tmp81 = tmp80.to(tl.float32)
    tmp82 = tmp81 - tmp47
    tmp83 = tmp82 + tmp5
    tmp84 = tl.full([1, 1], 66, tl.int64)
    tmp85 = tmp1 * tmp84
    tmp86 = tmp85 + tmp45
    tmp87 = tl.full([XBLOCK, YBLOCK], 4356, tl.int32)
    tmp88 = tmp86 + tmp87
    tmp89 = tmp86 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp86)
    tl.device_assert(((0 <= tmp90) & (tmp90 < 4356)) | ~(xmask), "index out of bounds: 0 <= tmp90 < 4356")
    tmp92 = (-1) + (((tmp90 // 66) % 66))
    tmp93 = tmp92.to(tl.int32)
    tmp94 = tmp93 >= tmp13
    tmp95 = tl.full([1, 1], 64, tl.int64)
    tmp96 = tmp93 < tmp95
    tmp97 = (-1) + ((tmp90 % 66))
    tmp98 = tmp97.to(tl.int32)
    tmp99 = tmp98 >= tmp13
    tmp100 = tmp98 < tmp95
    tmp101 = tmp94 & tmp96
    tmp102 = tmp101 & tmp99
    tmp103 = tmp102 & tmp100
    tmp104 = tl.load(in_ptr3 + (tl.broadcast_to((-65) + 64*(((tmp90 // 66) % 66)) + 4096*y3 + ((tmp90 % 66)), [XBLOCK, YBLOCK])), tmp103 & xmask, eviction_policy='evict_last', other=0.0)
    tmp105 = tmp8 * tmp84
    tmp106 = tmp105 + tmp51
    tmp107 = tmp106 + tmp87
    tmp108 = tmp106 < 0
    tmp109 = tl.where(tmp108, tmp107, tmp106)
    tl.device_assert(((0 <= tmp109) & (tmp109 < 4356)) | ~(xmask), "index out of bounds: 0 <= tmp109 < 4356")
    tmp111 = (-1) + (((tmp109 // 66) % 66))
    tmp112 = tmp111.to(tl.int32)
    tmp113 = tmp112 >= tmp13
    tmp114 = tmp112 < tmp95
    tmp115 = (-1) + ((tmp109 % 66))
    tmp116 = tmp115.to(tl.int32)
    tmp117 = tmp116 >= tmp13
    tmp118 = tmp116 < tmp95
    tmp119 = tmp113 & tmp114
    tmp120 = tmp119 & tmp117
    tmp121 = tmp120 & tmp118
    tmp122 = tl.load(in_ptr3 + (tl.broadcast_to((-65) + 64*(((tmp109 // 66) % 66)) + 4096*y3 + ((tmp109 % 66)), [XBLOCK, YBLOCK])), tmp121 & xmask, eviction_policy='evict_last', other=0.0)
    tmp123 = tmp40 * tmp84
    tmp124 = tmp123 + tmp80
    tmp125 = tmp124 + tmp87
    tmp126 = tmp124 < 0
    tmp127 = tl.where(tmp126, tmp125, tmp124)
    tl.device_assert(((0 <= tmp127) & (tmp127 < 4356)) | ~(xmask), "index out of bounds: 0 <= tmp127 < 4356")
    tmp129 = (-1) + (((tmp127 // 66) % 66))
    tmp130 = tmp129.to(tl.int32)
    tmp131 = tmp130 >= tmp13
    tmp132 = tmp130 < tmp95
    tmp133 = (-1) + ((tmp127 % 66))
    tmp134 = tmp133.to(tl.int32)
    tmp135 = tmp134 >= tmp13
    tmp136 = tmp134 < tmp95
    tmp137 = tmp131 & tmp132
    tmp138 = tmp137 & tmp135
    tmp139 = tmp138 & tmp136
    tmp140 = tl.load(in_ptr3 + (tl.broadcast_to((-65) + 64*(((tmp127 // 66) % 66)) + 4096*y3 + ((tmp127 % 66)), [XBLOCK, YBLOCK])), tmp139 & xmask, eviction_policy='evict_last', other=0.0)
    tmp141 = tmp28 * tmp84
    tmp142 = tmp141 + tmp68
    tmp143 = tmp142 + tmp87
    tmp144 = tmp142 < 0
    tmp145 = tl.where(tmp144, tmp143, tmp142)
    tl.device_assert(((0 <= tmp145) & (tmp145 < 4356)) | ~(xmask), "index out of bounds: 0 <= tmp145 < 4356")
    tmp147 = (-1) + (((tmp145 // 66) % 66))
    tmp148 = tmp147.to(tl.int32)
    tmp149 = tmp148 >= tmp13
    tmp150 = tmp148 < tmp95
    tmp151 = (-1) + ((tmp145 % 66))
    tmp152 = tmp151.to(tl.int32)
    tmp153 = tmp152 >= tmp13
    tmp154 = tmp152 < tmp95
    tmp155 = tmp149 & tmp150
    tmp156 = tmp155 & tmp153
    tmp157 = tmp156 & tmp154
    tmp158 = tl.load(in_ptr3 + (tl.broadcast_to((-65) + 64*(((tmp145 // 66) % 66)) + 4096*y3 + ((tmp145 % 66)), [XBLOCK, YBLOCK])), tmp157 & xmask, eviction_policy='evict_last', other=0.0)
    tmp159 = tmp6 * tmp49
    tmp160 = tmp159 * tmp104
    tmp161 = tmp11 * tmp54
    tmp162 = tmp161 * tmp122
    tmp163 = tmp160 + tmp162
    tmp164 = tmp31 * tmp71
    tmp165 = tmp164 * tmp158
    tmp166 = tmp163 + tmp165
    tmp167 = tmp43 * tmp83
    tmp168 = tmp167 * tmp140
    tmp169 = tmp166 + tmp168
    tmp171 = tl.sigmoid(tmp170)
    tmp172 = tmp169 * tmp171
    tl.store(out_ptr0 + (x1 + 9*y0), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + 9*y0), tmp11, xmask)
    tl.store(out_ptr2 + (x1 + 9*y0), tmp31, xmask)
    tl.store(out_ptr3 + (x1 + 9*y0), tmp43, xmask)
    tl.store(out_ptr4 + (x1 + 9*y0), tmp49, xmask)
    tl.store(out_ptr5 + (x1 + 9*y0), tmp54, xmask)
    tl.store(out_ptr6 + (x1 + 9*y0), tmp71, xmask)
    tl.store(out_ptr7 + (x1 + 9*y0), tmp83, xmask)
    tl.store(out_ptr8 + (x1 + 9*y0), tmp104, xmask)
    tl.store(out_ptr9 + (x1 + 9*y0), tmp122, xmask)
    tl.store(out_ptr10 + (x1 + 9*y0), tmp140, xmask)
    tl.store(out_ptr11 + (x1 + 9*y0), tmp158, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 9*y0), tmp172, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/my/cmynypjnkxcqofrdiomu6a5idolgkmb2p3xtsxo7v4onpnzn3mk5.py
# Topologically Sorted Source Nodes: [x_offset_6, x_offset_7], Original ATen: [aten.cat, aten.view]
# Source node to ATen node mapping:
#   x_offset_6 => cat_7
#   x_offset_7 => view_26
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_23, %view_24, %view_25], -1), kwargs = {})
#   %view_26 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_7, [4, 1, 192, 192]), kwargs = {})
triton_poi_fused_cat_view_6 = async_compile.triton('triton_poi_fused_cat_view_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_view_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 576)
    x1 = xindex // 576
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (9*((((x0) // 3) % 64)) + 576*x1 + (((x0) % 3))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 384, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (3 + 9*(((((-192) + x0) // 3) % 64)) + 576*x1 + ((((-192) + x0) % 3))), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (6 + 9*(((((-384) + x0) // 3) % 64)) + 576*x1 + ((((-384) + x0) % 3))), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(in_out_ptr0 + (x4), tmp16, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 9, 64, 64), (36864, 4096, 64, 1))
    assert_size_stride(primals_2, (4, 18, 64, 64), (73728, 4096, 64, 1))
    assert_size_stride(primals_3, (4, 1, 64, 64), (4096, 4096, 64, 1))
    assert_size_stride(primals_4, (32, 1, 3, 3), (9, 9, 3, 1))
    buf2 = empty_strided_cpu((18, ), (1, ), torch.int64)
    buf0 = reinterpret_tensor(buf2, (9, ), (1, ), 0)  # alias
    buf1 = reinterpret_tensor(buf2, (9, ), (1, ), 9)  # alias
    cpp_fused_cat_0(buf0, buf1)
    del buf0
    del buf1
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 18, 1, 1), (18, 1, 1, 1), torch.int64)
        buf3.copy_(reinterpret_tensor(buf2, (1, 18, 1, 1), (18, 1, 1, 1), 0), False)
        del buf2
    buf6 = empty_strided_cpu((1, 18, 64, 64), (73728, 4096, 64, 1), torch.int64)
    buf4 = reinterpret_tensor(buf6, (1, 9, 64, 64), (73728, 4096, 64, 1), 0)  # alias
    buf5 = reinterpret_tensor(buf6, (1, 9, 64, 64), (73728, 4096, 64, 1), 36864)  # alias
    cpp_fused_repeat_1(buf4, buf5)
    del buf4
    del buf5
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf7 = empty_strided_cuda((1, 18, 64, 64), (73728, 4096, 64, 1), torch.int64)
        buf7.copy_(buf6, False)
        del buf6
        buf28 = empty_strided_cuda((4, 64, 64, 9), (36864, 64, 1, 4096), torch.bool)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.ge, aten.le, aten.logical_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ge_le_logical_and_2.run(buf7, buf3, primals_2, buf28, 147456, grid=grid(147456), stream=stream0)
        buf29 = empty_strided_cuda((4, 64, 64, 9), (36864, 64, 1, 4096), torch.bool)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.ge, aten.le, aten.logical_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ge_le_logical_and_3.run(buf7, buf3, primals_2, buf29, 147456, grid=grid(147456), stream=stream0)
        buf8 = empty_strided_cuda((4, 64, 64, 18), (73728, 1152, 18, 1), torch.float32)
        buf9 = empty_strided_cuda((4, 64, 64, 18), (73728, 1152, 18, 1), torch.float32)
        buf10 = empty_strided_cuda((4, 64, 64, 18), (73728, 1152, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2, cat_3, p_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf7, buf3, primals_2, buf8, buf9, buf10, 294912, grid=grid(294912), stream=stream0)
        del buf3
        del buf7
        del primals_2
        buf11 = empty_strided_cuda((4, 64, 64, 9), (36864, 576, 9, 1), torch.float32)
        buf13 = empty_strided_cuda((4, 64, 64, 9), (36864, 576, 9, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 64, 64, 9), (36864, 576, 9, 1), torch.float32)
        buf17 = empty_strided_cuda((4, 64, 64, 9), (36864, 576, 9, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 64, 64, 9), (36864, 576, 9, 1), torch.float32)
        buf14 = empty_strided_cuda((4, 64, 64, 9), (36864, 576, 9, 1), torch.float32)
        buf16 = empty_strided_cuda((4, 64, 64, 9), (36864, 576, 9, 1), torch.float32)
        buf18 = empty_strided_cuda((4, 64, 64, 9), (36864, 576, 9, 1), torch.float32)
        buf19 = empty_strided_cuda((4, 1, 36864), (36864, 36864, 1), torch.float32)
        buf20 = empty_strided_cuda((4, 1, 36864), (36864, 36864, 1), torch.float32)
        buf22 = empty_strided_cuda((4, 1, 36864), (36864, 36864, 1), torch.float32)
        buf21 = empty_strided_cuda((4, 1, 36864), (36864, 36864, 1), torch.float32)
        buf23 = empty_strided_cuda((4, 1, 64, 64, 9), (36864, 147456, 576, 9, 1), torch.float32)
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [type_as, sub, add_3, type_as_1, sub_1, add_4, type_as_2, sub_2, sub_3, type_as_3, sub_4, sub_5, type_as_4, sub_6, add_5, type_as_5, sub_7, sub_8, type_as_6, sub_9, sub_10, type_as_7, sub_11, add_6, gather, gather_1, gather_2, gather_3, mul_8, mul_9, add_11, mul_10, add_12, mul_11, x_offset_4, x_offset_5], Original ATen: [aten._to_copy, aten.sub, aten.add, aten.rsub, aten.gather, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_gather_mul_rsub_sub_5.run(buf24, buf8, buf10, buf9, primals_3, primals_1, buf11, buf13, buf15, buf17, buf12, buf14, buf16, buf18, buf19, buf20, buf22, buf21, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del buf10
        del buf8
        del buf9
        del primals_3
        buf25 = empty_strided_cuda((4, 1, 64, 576), (36864, 1, 576, 1), torch.float32)
        buf26 = reinterpret_tensor(buf25, (4, 1, 192, 192), (36864, 36864, 192, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_offset_6, x_offset_7], Original ATen: [aten.cat, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_view_6.run(buf26, buf24, 147456, grid=grid(147456), stream=stream0)
        del buf24
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_4, stride=(3, 3), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 32, 64, 64), (131072, 4096, 64, 1))
    return (buf27, primals_1, primals_4, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, reinterpret_tensor(buf19, (4, 1, 64, 64, 9), (36864, 36864, 576, 9, 1), 0), reinterpret_tensor(buf20, (4, 1, 64, 64, 9), (36864, 36864, 576, 9, 1), 0), reinterpret_tensor(buf21, (4, 1, 64, 64, 9), (36864, 36864, 576, 9, 1), 0), reinterpret_tensor(buf22, (4, 1, 64, 64, 9), (36864, 36864, 576, 9, 1), 0), buf26, buf28, buf29, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 9, 64, 64), (36864, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 18, 64, 64), (73728, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 1, 64, 64), (4096, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
