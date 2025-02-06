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
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(4L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = 1L + x1;
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            out_ptr0[static_cast<int64_t>(x2 + 4L*x1 + 16L*x0)] = tmp1;
                        }
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(36L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = 1L + x1;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        out_ptr1[static_cast<int64_t>(x1 + 4L*x0)] = tmp1;
                    }
                }
            }
        }
    }
}
''')


# kernel path: inductor_cache/so/csotqvaysvdwfhjul5nteivcwxbt4chp4kwmrodiaerek4y72yaa.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.ge, aten.le, aten.logical_and]
# Source node to ATen node mapping:
# Graph fragment:
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%slice_10, 0), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%slice_10, 5), kwargs = {})
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
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_le_logical_and_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ge_le_logical_and_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex % 144)
    x1 = ((xindex // 16) % 9)
    x2 = xindex // 144
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (144 + x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (9 + x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (144 + x3 + 288*x2), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 0.0
    tmp8 = tmp6 >= tmp7
    tmp9 = 5.0
    tmp10 = tmp6 <= tmp9
    tmp11 = tmp8 & tmp10
    tl.store(out_ptr0 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ij/ciju4qkx73xr3v3amhcdc4vaujxdtkxivysexgx6vs36nybnpqhs.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.ge, aten.le, aten.logical_and]
# Source node to ATen node mapping:
# Graph fragment:
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%slice_9, 0), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%slice_9, 5), kwargs = {})
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
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_le_logical_and_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ge_le_logical_and_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex % 144)
    x1 = ((xindex // 16) % 9)
    x2 = xindex // 144
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x3 + 288*x2), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 0.0
    tmp8 = tmp6 >= tmp7
    tmp9 = 5.0
    tmp10 = tmp6 <= tmp9
    tmp11 = tmp8 & tmp10
    tl.store(out_ptr0 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w4/cw46z3m2ujnmiv3fq65lc2spb6qkja3k4ktshxrfn7mt6wadyxj3.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_2, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_4 = async_compile.triton('triton_poi_fused_constant_pad_nd_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1 + 16*x2), tmp10 & xmask, other=0.0)
    tl.store(out_ptr0 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k3/ck3ovkkjalo4tfsafhnapqicxsq46dnfek2m2h73lbsuyyms33j5.py
# Topologically Sorted Source Nodes: [cat_2, cat_3, p_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
#   cat_3 => cat_3
#   p_2 => cat_6
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max, %clamp_max_1], -1), kwargs = {})
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_2, %clamp_max_3], -1), kwargs = {})
#   %cat_6 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%clamp_max_4, %clamp_max_5], -1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 18)
    x1 = ((xindex // 18) % 16)
    x2 = xindex // 288
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 9, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 16*(x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.load(in_ptr2 + (x1 + 16*(x0) + 288*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.floor(tmp11)
    tmp13 = 0.0
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = 5.0
    tmp16 = triton_helpers.minimum(tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 18, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (144 + x1 + 16*((-9) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (9 + ((-9) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 + tmp25
    tmp27 = tl.load(in_ptr2 + (144 + x1 + 16*((-9) + x0) + 288*x2), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.floor(tmp28)
    tmp30 = 0.0
    tmp31 = triton_helpers.maximum(tmp29, tmp30)
    tmp32 = 5.0
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
    tl.store(out_ptr0 + (x4), tmp36, xmask)
    tl.store(out_ptr1 + (x4), tmp49, xmask)
    tl.store(out_ptr2 + (x4), tmp58, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/po/cpoyhqj3bfcj25o3engr2hjef7x37a6i5y74tin65trbmrbw4ium.py
# Topologically Sorted Source Nodes: [mul_4, index], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   index => add_7
#   mul_4 => mul_4
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_5, 6), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %slice_8), kwargs = {})
triton_poi_fused_add_mul_6 = async_compile.triton('triton_poi_fused_add_mul_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 9)
    x1 = xindex // 9
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 18*x1), xmask)
    tmp4 = tl.load(in_ptr0 + (9 + x0 + 18*x1), xmask)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1], 6, tl.int64)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tmp3 + tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wp/cwprzssukra6u2uzivuuih3omlj3f6f5yybqjuto6reswsle6sof.py
# Topologically Sorted Source Nodes: [q_lb], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   q_lb => cat_4
# Graph fragment:
#   %cat_4 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_5, %slice_6], -1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 18)
    x1 = xindex // 18
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 9, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (18*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 18, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr1 + (9 + 18*x1 + ((-9) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp9, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp8, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g4/cg4pc7ap5v6yjpyyhz3zkyfmn7obvcah43zpo6kh3ptlcyjplzp5.py
# Topologically Sorted Source Nodes: [mul_8, mul_9, add_11, mul_10, add_12, mul_11, x_offset_4], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_11 => add_11
#   add_12 => add_12
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_8 => mul_8
#   mul_9 => mul_9
#   x_offset_4 => add_13
# Graph fragment:
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_4, %view_13), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_16), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_6, %view_19), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_22), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mul_11), kwargs = {})
triton_poi_fused_add_mul_8 = async_compile.triton('triton_poi_fused_add_mul_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 9)
    x4 = xindex // 576
    x5 = ((xindex // 9) % 16)
    x7 = (xindex % 144)
    x8 = xindex // 144
    x9 = xindex
    x1 = ((xindex // 9) % 4)
    x2 = ((xindex // 36) % 4)
    tmp0 = tl.load(in_ptr0 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x7 + 144*x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x7 + 144*x4), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr6 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr6 + (x0 + 18*x1 + 72*x2 + 72*((x0 + 9*x1) // 36) + 288*x4), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr6 + (9 + x0 + 18*x1 + 72*x2 + 72*((x0 + 9*x1) // 36) + 288*x4), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr7 + (x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr7 + (9 + x0 + 18*x5 + 288*x4), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr7 + (x0 + 18*x1 + 72*x2 + 72*((x0 + 9*x1) // 36) + 288*x4), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr7 + (9 + x0 + 18*x1 + 72*x2 + 72*((x0 + 9*x1) // 36) + 288*x4), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 + tmp5
    tmp13 = tmp6 * tmp12
    tmp15 = tl.full([XBLOCK], 36, tl.int32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp14 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp14)
    tl.device_assert(((0 <= tmp18) & (tmp18 < 36)) | ~(xmask), "index out of bounds: 0 <= tmp18 < 36")
    tmp20 = tl.load(in_ptr3 + (tmp18 + 36*x8), xmask, eviction_policy='evict_last')
    tmp21 = tmp13 * tmp20
    tmp23 = tmp22.to(tl.int64)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 - tmp3
    tmp26 = tmp5 - tmp25
    tmp28 = tmp27.to(tl.int64)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 - tmp10
    tmp31 = tmp5 - tmp30
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33 + tmp15
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tl.device_assert(((0 <= tmp36) & (tmp36 < 36)) | ~(xmask), "index out of bounds: 0 <= tmp36 < 36")
    tmp38 = tl.load(in_ptr3 + (tmp36 + 36*x8), xmask, eviction_policy='evict_last')
    tmp39 = tmp32 * tmp38
    tmp40 = tmp21 + tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 - tmp3
    tmp44 = tmp43 + tmp5
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46 - tmp10
    tmp48 = tmp5 - tmp47
    tmp49 = tmp44 * tmp48
    tmp51 = tl.full([1], 6, tl.int64)
    tmp52 = tmp50 * tmp51
    tmp54 = tmp52 + tmp53
    tmp55 = tmp54 + tmp15
    tmp56 = tmp54 < 0
    tmp57 = tl.where(tmp56, tmp55, tmp54)
    tl.device_assert(((0 <= tmp57) & (tmp57 < 36)) | ~(xmask), "index out of bounds: 0 <= tmp57 < 36")
    tmp59 = tl.load(in_ptr3 + (tmp57 + 36*x8), xmask, eviction_policy='evict_last')
    tmp60 = tmp49 * tmp59
    tmp61 = tmp40 + tmp60
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp63 - tmp3
    tmp65 = tmp5 - tmp64
    tmp67 = tmp66.to(tl.float32)
    tmp68 = tmp67 - tmp10
    tmp69 = tmp68 + tmp5
    tmp70 = tmp65 * tmp69
    tmp72 = tmp71 * tmp51
    tmp74 = tmp72 + tmp73
    tmp75 = tmp74 + tmp15
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tl.device_assert(((0 <= tmp77) & (tmp77 < 36)) | ~(xmask), "index out of bounds: 0 <= tmp77 < 36")
    tmp79 = tl.load(in_ptr3 + (tmp77 + 36*x8), xmask, eviction_policy='evict_last')
    tmp80 = tmp70 * tmp79
    tmp81 = tmp61 + tmp80
    tl.store(in_out_ptr0 + (x9), tmp81, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5w/c5wn3te7rw2vaznglj66nqevxnpzjmd4neuacl5tdnnmtilnk4ut.py
# Topologically Sorted Source Nodes: [x_offset_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_offset_5 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_23, %view_24, %view_25], -1), kwargs = {})
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 36)
    x1 = xindex // 36
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (9*((((x0) // 3) % 4)) + 36*x1 + (((x0) % 3))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 24, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (3 + 9*(((((-12) + x0) // 3) % 4)) + 36*x1 + ((((-12) + x0) % 3))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 36, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (6 + 9*(((((-24) + x0) // 3) % 4)) + 36*x1 + ((((-24) + x0) % 3))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v3/cv3sdya77p6f7o2mfevqw75jjzp7kb5werluvjfemktqlhkch4o2.py
# Topologically Sorted Source Nodes: [x_offset_5, x_offset_6], Original ATen: [aten.cat, aten.view]
# Source node to ATen node mapping:
#   x_offset_5 => cat_7
#   x_offset_6 => view_26
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_23, %view_24, %view_25], -1), kwargs = {})
#   %view_26 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_7, [4, 4, 12, 12]), kwargs = {})
triton_poi_fused_cat_view_10 = async_compile.triton('triton_poi_fused_cat_view_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_view_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_view_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 18, 4, 4), (288, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, 4, 3, 3), (36, 9, 3, 1))
    buf3 = empty_strided_cpu((18, ), (1, ), torch.int64)
    buf1 = reinterpret_tensor(buf3, (9, ), (1, ), 0)  # alias
    buf2 = reinterpret_tensor(buf3, (9, ), (1, ), 9)  # alias
    cpp_fused_cat_0(buf1, buf2)
    del buf1
    del buf2
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((1, 18, 1, 1), (18, 1, 1, 1), torch.int64)
        buf4.copy_(reinterpret_tensor(buf3, (1, 18, 1, 1), (18, 1, 1, 1), 0), False)
        del buf3
    buf7 = empty_strided_cpu((1, 18, 4, 4), (288, 16, 4, 1), torch.int64)
    buf5 = reinterpret_tensor(buf7, (1, 9, 4, 4), (288, 16, 4, 1), 0)  # alias
    buf6 = reinterpret_tensor(buf7, (1, 9, 4, 4), (288, 16, 4, 1), 144)  # alias
    cpp_fused_repeat_1(buf5, buf6)
    del buf5
    del buf6
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = empty_strided_cuda((1, 18, 4, 4), (288, 16, 4, 1), torch.int64)
        buf8.copy_(buf7, False)
        del buf7
        buf21 = empty_strided_cuda((4, 4, 4, 9), (144, 4, 1, 16), torch.bool)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.ge, aten.le, aten.logical_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ge_le_logical_and_2.run(buf8, buf4, primals_1, buf21, 576, grid=grid(576), stream=stream0)
        buf22 = empty_strided_cuda((4, 4, 4, 9), (144, 4, 1, 16), torch.bool)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.ge, aten.le, aten.logical_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ge_le_logical_and_3.run(buf8, buf4, primals_1, buf22, 576, grid=grid(576), stream=stream0)
        buf0 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_4.run(primals_2, buf0, 576, grid=grid(576), stream=stream0)
        del primals_2
        buf9 = empty_strided_cuda((4, 4, 4, 18), (288, 72, 18, 1), torch.float32)
        buf10 = empty_strided_cuda((4, 4, 4, 18), (288, 72, 18, 1), torch.float32)
        buf13 = empty_strided_cuda((4, 4, 4, 18), (288, 72, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2, cat_3, p_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf8, buf4, primals_1, buf9, buf10, buf13, 1152, grid=grid(1152), stream=stream0)
        del buf4
        del buf8
        del primals_1
        buf14 = empty_strided_cuda((4, 4, 4, 9), (144, 36, 9, 1), torch.int64)
        # Topologically Sorted Source Nodes: [mul_4, index], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf9, buf14, 576, grid=grid(576), stream=stream0)
        buf15 = empty_strided_cuda((4, 4, 4, 9), (144, 36, 9, 1), torch.int64)
        # Topologically Sorted Source Nodes: [mul_5, index_2], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf10, buf15, 576, grid=grid(576), stream=stream0)
        buf11 = empty_strided_cuda((4, 4, 4, 18), (288, 72, 18, 1), torch.int64)
        # Topologically Sorted Source Nodes: [q_lb], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf9, buf10, buf11, 1152, grid=grid(1152), stream=stream0)
        buf12 = empty_strided_cuda((4, 4, 4, 18), (288, 72, 18, 1), torch.int64)
        # Topologically Sorted Source Nodes: [q_rt], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf10, buf9, buf12, 1152, grid=grid(1152), stream=stream0)
        buf16 = empty_strided_cuda((4, 4, 4, 4, 9), (576, 144, 36, 9, 1), torch.float32)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [mul_8, mul_9, add_11, mul_10, add_12, mul_11, x_offset_4], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_8.run(buf17, buf9, buf13, buf14, buf0, buf10, buf15, buf11, buf12, 2304, grid=grid(2304), stream=stream0)
        buf18 = empty_strided_cuda((4, 4, 4, 36), (576, 144, 36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_offset_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_9.run(buf17, buf18, 2304, grid=grid(2304), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (4, 4, 12, 12), (576, 144, 12, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_offset_5, x_offset_6], Original ATen: [aten.cat, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_view_10.run(buf18, buf19, 2304, grid=grid(2304), stream=stream0)
        del buf18
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_3, stride=(3, 3), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 4, 4, 4), (64, 16, 4, 1))
    return (buf20, primals_3, buf0, buf9, buf10, buf11, buf12, buf13, reinterpret_tensor(buf14, (4, 1, 4, 4, 9), (144, 144, 36, 9, 1), 0), reinterpret_tensor(buf15, (4, 1, 4, 4, 9), (144, 144, 36, 9, 1), 0), buf19, buf21, buf22, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 18, 4, 4), (288, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
