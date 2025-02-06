# AOT ID: ['8_forward']
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


# kernel path: inductor_cache/lc/clcniylhrhwxuijq3r6ub3nbq2x3nhnnw7c4poe2cgdwg5o3maf3.py
# Topologically Sorted Source Nodes: [pool], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   pool => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%primals_1, [4, 4], [4, 4], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tl.store(out_ptr0 + (x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mk/cmk3ojx23ljjadmvpiuqqwpekpavtra7qptxrryznmhva4mdoc4k.py
# Topologically Sorted Source Nodes: [pool_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten._unsafe_index, aten.sub, aten.add]
# Source node to ATen node mapping:
#   pool_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_2, add_3, add_4, clamp_max_2, clamp_max_3, clamp_min, clamp_min_2, clamp_min_3, convert_element_type, convert_element_type_1, convert_element_type_3, iota, mul, mul_2, mul_3, mul_4, sub, sub_1, sub_2, sub_3, sub_4
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul, 0.0), kwargs = {})
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min, torch.int64), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%getitem, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %clamp_max_2), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %clamp_max_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_3), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %add_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_3), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_4), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_1 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 - tmp0
    tmp2 = 0.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tmp4 - tmp4
    tmp6 = tmp5 * tmp2
    tmp7 = tmp4 + tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d3/cd3xmssqpxjzi2dcf7sx6tezzye33plywl5zf3iy7ykosm46qdkp.py
# Topologically Sorted Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   leaky_relu => gt, mul_5, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul_5), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%where, 0), kwargs = {})
triton_poi_fused_leaky_relu_leaky_relu_backward_2 = async_compile.triton('triton_poi_fused_leaky_relu_leaky_relu_backward_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lh/clhlyjkrc4fzpii667avhniisrbhaxypf6ckoqmwjstzcyajbunw.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_1, %where, %where_1], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 8)
    x0 = (xindex % 16)
    x2 = xindex // 128
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 6, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-4) + x1) + 32*x2), tmp9 & xmask, other=0.0)
    tmp11 = 0.0
    tmp12 = tmp10 > tmp11
    tmp13 = 0.1
    tmp14 = tmp10 * tmp13
    tmp15 = tl.where(tmp12, tmp10, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp9, tmp15, tmp16)
    tmp18 = tmp0 >= tmp7
    tmp19 = tl.full([1], 8, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + (x0 + 16*((-6) + x1) + 32*x2), tmp18 & xmask, other=0.0)
    tmp22 = 0.0
    tmp23 = tmp21 > tmp22
    tmp24 = 0.1
    tmp25 = tmp21 * tmp24
    tmp26 = tl.where(tmp23, tmp21, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp18, tmp26, tmp27)
    tmp29 = tl.where(tmp9, tmp17, tmp28)
    tmp30 = tl.where(tmp4, tmp5, tmp29)
    tl.store(out_ptr0 + (x3), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nz/cnzdxii6uihwwmlw4l4xbxnqxa4jzzdrtjc4a4gc5jb5um6uiwmo.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   input_1 => gt_2, mul_12, where_2
# Graph fragment:
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 0.1), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_2, %mul_12), kwargs = {})
triton_poi_fused_leaky_relu_4 = async_compile.triton('triton_poi_fused_leaky_relu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_4(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.1
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_4, (4, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_5, (4, 4, 1, 1), (4, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [pool], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0.run(primals_1, buf0, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pool_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten._unsafe_index, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_1.run(buf0, buf1, 256, grid=grid(256), stream=stream0)
        del buf0
        # Topologically Sorted Source Nodes: [conv], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 2, 4, 4), (32, 16, 4, 1))
        buf9 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_2.run(buf2, buf9, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf1, primals_3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 2, 4, 4), (32, 16, 4, 1))
        buf8 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [leaky_relu_1], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_2.run(buf3, buf8, 128, grid=grid(128), stream=stream0)
        buf4 = empty_strided_cuda((4, 8, 4, 4), (128, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(primals_1, buf2, buf3, buf4, 512, grid=grid(512), stream=stream0)
        del buf2
        del buf3
        del primals_1
        # Topologically Sorted Source Nodes: [conv_2], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 4, 4, 4), (64, 16, 4, 1))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_4.run(buf6, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [conv_3], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 4, 4, 4), (64, 16, 4, 1))
    return (buf7, primals_2, primals_3, primals_4, primals_5, buf1, buf4, buf6, buf8, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
