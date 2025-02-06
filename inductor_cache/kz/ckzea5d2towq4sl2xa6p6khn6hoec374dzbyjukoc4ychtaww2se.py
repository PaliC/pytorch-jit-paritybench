# AOT ID: ['47_forward']
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


# kernel path: inductor_cache/iy/ciymdro33goyaev5rkzn6uanexcrem23pzme4jwo3ej4cupgyd6f.py
# Topologically Sorted Source Nodes: [w], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   w => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %primals_3), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = in_ptr1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2a/c2aodogecoxi5ymah54vdp6gurlruuyfzuzfrxz2gs7zyfxbdhqt.py
# Topologically Sorted Source Nodes: [out, square_1, mean_1, add_2, rsqrt_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   add_2 => add_2
#   mean_1 => mean_1
#   out => add_1
#   rsqrt_1 => rsqrt_1
#   square_1 => pow_2
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %view), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [1], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
triton_poi_fused_add_mean_pow_rsqrt_1 = async_compile.triton('triton_poi_fused_add_mean_pow_rsqrt_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_pow_rsqrt_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_pow_rsqrt_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (1))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (2))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (3))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3 * tmp3
    tmp8 = tmp5 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tmp4 + tmp9
    tmp14 = tmp11 + tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tmp10 + tmp15
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tmp16 + tmp21
    tmp23 = 4.0
    tmp24 = tmp22 / tmp23
    tmp25 = 1.1920928955078125e-07
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tl.store(out_ptr0 + (x0), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fs/cfs2xbk7ph5yakb5cptwn5d2v2ms7byzu2etwqddwzqzyetgdbtr.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_2 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul, %mul_2], 1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (4*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 * tmp6
    tmp8 = tl.load(in_ptr0 + (1 + 4*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp11 = tl.load(in_ptr0 + (2 + 4*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tl.load(in_ptr0 + (3 + 4*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = 4.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1.1920928955078125e-07
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp5 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 8, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr1 + (4*x1 + ((-4) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr2 + ((-4) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.load(in_ptr3 + (x1), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp25, tmp32, tmp33)
    tmp35 = tl.where(tmp4, tmp24, tmp34)
    tl.store(out_ptr0 + (x2), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ma/cma6e4dng2uxi3dyr65cvibzlywh6tokao7a4dbptekvgsksg4g7.py
# Topologically Sorted Source Nodes: [w_1, t_1], Original ATen: [aten.mul, aten.t]
# Source node to ATen node mapping:
#   t_1 => permute_1
#   w_1 => mul_3
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %primals_7), kwargs = {})
#   %permute_1 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%mul_3, [1, 0]), kwargs = {})
triton_poi_fused_mul_t_3 = async_compile.triton('triton_poi_fused_mul_t_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_t_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_t_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = in_ptr1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


cpp_fused_4 = async_compile.cpp_pybinding(['const double*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const double* in_ptr0,
                       float* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = c10::convert<float>(tmp0);
                out_ptr0[static_cast<int64_t>(0L)] = tmp1;
            }
        }
    }
}
''')


# kernel path: inductor_cache/ta/ctapc43cqech5dwnqk6d564sorxp7rlvprwnwqhb66doxstljz6i.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_4 => add_3
#   x_5 => gt, mul_5, where
# Graph fragment:
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, %view_1), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.2), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_3, %mul_5), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %convert_element_type_default), kwargs = {})
triton_poi_fused_add_leaky_relu_5 = async_compile.triton('triton_poi_fused_add_leaky_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': 'fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = in_ptr1
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.2
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dr/cdrgtcjajbldnr4olaoofegq5uqk3vvnzom2hkry4ut6svczd7dl.py
# Topologically Sorted Source Nodes: [x_32, x_33], Original ATen: [aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   x_32 => add_10
#   x_33 => gt_7
# Graph fragment:
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_8, %view_8), kwargs = {})
#   %gt_7 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_10, 0), kwargs = {})
triton_poi_fused_add_leaky_relu_6 = async_compile.triton('triton_poi_fused_add_leaky_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/iz/ciznuq2tjrkpmvob2xp3awd6u7fkqjw6rccy5m7up4sltac2fk6j.py
# Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   x_35 => repeat
# Graph fragment:
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze, [1, 4, 1]), kwargs = {})
triton_poi_fused_repeat_7 = async_compile.triton('triton_poi_fused_repeat_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = in_ptr3
    tmp3 = 0.01
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 + tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp8 = tl.where(tmp0, tmp5, tmp7)
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (), ())
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, 4), (4, 1))
    assert_size_stride(primals_6, (4, 8), (8, 1))
    assert_size_stride(primals_7, (), ())
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (), ())
    assert_size_stride(primals_10, (4, 4), (4, 1))
    assert_size_stride(primals_11, (), ())
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (4, 4), (4, 1))
    assert_size_stride(primals_14, (), ())
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, 4), (4, 1))
    assert_size_stride(primals_17, (), ())
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, 4), (4, 1))
    assert_size_stride(primals_20, (), ())
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, 4), (4, 1))
    assert_size_stride(primals_23, (), ())
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, 4), (4, 1))
    assert_size_stride(primals_26, (), ())
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, 4), (4, 1))
    assert_size_stride(primals_29, (), ())
    assert_size_stride(primals_30, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [w], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_2, primals_3.item(), buf0, 16, grid=grid(16), stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.mm]
        extern_kernels.mm(primals_5, reinterpret_tensor(buf0, (4, 4), (1, 4), 0), out=buf1)
        buf2 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [out, square_1, mean_1, add_2, rsqrt_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_pow_rsqrt_1.run(buf1, primals_4, buf2, 4, grid=grid(4), stream=stream0)
        buf3 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(primals_1, buf1, primals_4, buf2, buf3, 32, grid=grid(32), stream=stream0)
        del buf2
        del primals_1
        buf4 = empty_strided_cuda((8, 4), (1, 8), torch.float32)
        # Topologically Sorted Source Nodes: [w_1, t_1], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_t_3.run(primals_6, primals_7.item(), buf4, 32, grid=grid(32), stream=stream0)
        del primals_6
        buf5 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, buf4, out=buf5)
    buf7 = empty_strided_cpu((), (), torch.float32)
    cpp_fused_4(primals_9, buf7)
    del primals_9
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf8 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_5.run(buf8, primals_8, buf7.item(), buf6, 16, grid=grid(16), stream=stream0)
        del primals_8
        buf9 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [w_2, t_2], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_10, primals_11.item(), buf9, 16, grid=grid(16), stream=stream0)
        del primals_10
        buf10 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, buf9, out=buf10)
        buf11 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_5.run(buf12, primals_12, buf7.item(), buf11, 16, grid=grid(16), stream=stream0)
        del primals_12
        buf13 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [w_3, t_3], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_13, primals_14.item(), buf13, 16, grid=grid(16), stream=stream0)
        del primals_13
        buf14 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf12, buf13, out=buf14)
        buf15 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_12, x_13], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_5.run(buf16, primals_15, buf7.item(), buf15, 16, grid=grid(16), stream=stream0)
        del primals_15
        buf17 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [w_4, t_4], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_16, primals_17.item(), buf17, 16, grid=grid(16), stream=stream0)
        del primals_16
        buf18 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, buf17, out=buf18)
        buf19 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf20 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_5.run(buf20, primals_18, buf7.item(), buf19, 16, grid=grid(16), stream=stream0)
        del primals_18
        buf21 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [w_5, t_5], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_19, primals_20.item(), buf21, 16, grid=grid(16), stream=stream0)
        del primals_19
        buf22 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf20, buf21, out=buf22)
        buf23 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_20, x_21], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_5.run(buf24, primals_21, buf7.item(), buf23, 16, grid=grid(16), stream=stream0)
        del primals_21
        buf25 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [w_6, t_6], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_22, primals_23.item(), buf25, 16, grid=grid(16), stream=stream0)
        del primals_22
        buf26 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf24, buf25, out=buf26)
        buf27 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_24, x_25], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_5.run(buf28, primals_24, buf7.item(), buf27, 16, grid=grid(16), stream=stream0)
        del primals_24
        buf29 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [w_7, t_7], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_25, primals_26.item(), buf29, 16, grid=grid(16), stream=stream0)
        del primals_25
        buf30 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf28, buf29, out=buf30)
        buf31 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_5.run(buf32, primals_27, buf7.item(), buf31, 16, grid=grid(16), stream=stream0)
        del primals_27
        buf33 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [w_8, t_8], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(primals_28, primals_29.item(), buf33, 16, grid=grid(16), stream=stream0)
        del primals_28
        buf34 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.mm]
        extern_kernels.mm(buf32, buf33, out=buf34)
        buf35 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_32, x_33], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_6.run(buf34, primals_30, buf35, 16, grid=grid(16), stream=stream0)
        buf36 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_7.run(buf35, buf34, primals_30, buf7.item(), buf36, 64, grid=grid(64), stream=stream0)
        del buf34
        del primals_30
    return (buf36, primals_3, primals_4, primals_5, primals_7, primals_11, primals_14, primals_17, primals_20, primals_23, primals_26, primals_29, buf1, buf3, buf6, buf7, buf8, buf11, buf12, buf15, buf16, buf19, buf20, buf23, buf24, buf27, buf28, buf31, buf32, buf35, reinterpret_tensor(buf33, (4, 4), (4, 1), 0), reinterpret_tensor(buf29, (4, 4), (4, 1), 0), reinterpret_tensor(buf25, (4, 4), (4, 1), 0), reinterpret_tensor(buf21, (4, 4), (4, 1), 0), reinterpret_tensor(buf17, (4, 4), (4, 1), 0), reinterpret_tensor(buf13, (4, 4), (4, 1), 0), reinterpret_tensor(buf9, (4, 4), (4, 1), 0), reinterpret_tensor(buf4, (4, 8), (8, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_10 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
