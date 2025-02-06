# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/es/ceshwhmgere7sl6jri3ymrm3ky6rubqzbixtpt4zbseod3moknf2.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=7] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ob/cob7cmmx3k4n3jlexxqzlx3kpgmsfh2om5m5qgwrqfe6wpsbafib.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x_1 => add_1, clamp_max
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 1), kwargs = {})
#   %clamp_max : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_1, 3), kwargs = {})
triton_poi_fused_add_clamp_1 = async_compile.triton('triton_poi_fused_add_clamp_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dr/cdrfnhw4uwdyuoy26jrbqbkokk3byae2rnhusmswxp7gazhgrosy.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   x_1 => add, clamp_max_3, clamp_min, clamp_min_3, convert_element_type, iota, mul, sub, sub_3
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.5), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %clamp_min : [num_users=4] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_5), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_2 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fr/cfrprzykwdke6obwl2iqc6mij33te64qc2gq4ll7kyaqhukl5ttb.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_10, add_11, add_12, add_6, add_7, add_8, add_9, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_10, sub_12, sub_4, sub_5, sub_6, sub_7, sub_9
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %convert_element_type_1, %convert_element_type_3, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %convert_element_type_1, %convert_element_type_3, %clamp_max_2]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %convert_element_type_1, %clamp_max_1, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %convert_element_type_1, %clamp_max_1, %clamp_max_2]), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %clamp_max, %convert_element_type_3, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %clamp_max, %convert_element_type_3, %clamp_max_2]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %clamp_max, %clamp_max_1, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %clamp_max, %clamp_max_1, %clamp_max_2]), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_3), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_3), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %clamp_max_3), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_4), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_5), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %clamp_max_3), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_6), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %add_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %clamp_max_4), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_7), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %add_8), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %clamp_max_4), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mul_8), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %add_10), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %clamp_max_5), kwargs = {})
#   %add_12 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_9), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_3 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 8)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x3 = xindex // 512
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp10 = tmp9 + tmp1
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tmp13 = tl.load(in_ptr3 + (tmp12 + 4*tmp8 + 16*tmp4 + 64*x3), None, eviction_policy='evict_last')
    tmp15 = tmp14 + tmp1
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr3 + (tmp17 + 4*tmp8 + 16*tmp4 + 64*x3), None, eviction_policy='evict_last')
    tmp19 = tmp18 - tmp13
    tmp21 = tmp19 * tmp20
    tmp22 = tmp13 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tmp27 = tl.load(in_ptr3 + (tmp12 + 4*tmp8 + 16*tmp26 + 64*x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (tmp17 + 4*tmp8 + 16*tmp26 + 64*x3), None, eviction_policy='evict_last')
    tmp29 = tmp28 - tmp27
    tmp30 = tmp29 * tmp20
    tmp31 = tmp27 + tmp30
    tmp33 = tmp32 + tmp1
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tmp36 = tl.load(in_ptr3 + (tmp12 + 4*tmp35 + 16*tmp26 + 64*x3), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (tmp17 + 4*tmp35 + 16*tmp26 + 64*x3), None, eviction_policy='evict_last')
    tmp38 = tmp37 - tmp36
    tmp39 = tmp38 * tmp20
    tmp40 = tmp36 + tmp39
    tmp41 = tmp40 - tmp31
    tmp43 = tmp41 * tmp42
    tmp44 = tl.load(in_ptr3 + (tmp12 + 4*tmp35 + 16*tmp4 + 64*x3), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr3 + (tmp17 + 4*tmp35 + 16*tmp4 + 64*x3), None, eviction_policy='evict_last')
    tmp46 = tmp45 - tmp44
    tmp47 = tmp46 * tmp20
    tmp48 = tmp44 + tmp47
    tmp49 = tmp48 - tmp22
    tmp50 = tmp49 * tmp42
    tmp51 = tmp31 + tmp43
    tmp52 = tmp22 + tmp50
    tmp53 = tmp52 - tmp51
    tmp55 = tmp53 * tmp54
    tmp56 = tmp51 + tmp55
    tl.store(in_out_ptr0 + (x6), tmp56, None)
''', device_str='cuda')


# kernel path: inductor_cache/2h/c2hustzc4gqzdmsbxxab4xbl2lppsti4aj7hhg5m5gcq3mv23gzc.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_14, mul_11, mul_12, sub_13
#   out_2 => relu
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_2), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_5), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %unsqueeze_8), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %unsqueeze_11), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/tn/ctnezepnqssy4hpa6gfn3cmxcy6ertzcjlhpvbin3eyolr4btadv.py
# Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_4 => add_16, mul_14, mul_15, sub_14
#   out_5 => add_17
#   out_6 => relu_1
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_14), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_17), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %unsqueeze_20), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %unsqueeze_23), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %add_12), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 512) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/ry/crytm6vbwto7kmothtotsnkghzevhg6p7gdixd7csfxnqnc4cdz3.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_3 => convert_element_type_23
# Graph fragment:
#   %convert_element_type_23 : [num_users=7] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.int64), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ha/cha3mgcfv3xpvq6nltf6s2mv5o2hzyfcf2gamts26jpqgb6pbhhc.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x_3 => add_34, clamp_max_6
# Graph fragment:
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_23, 1), kwargs = {})
#   %clamp_max_6 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_34, 7), kwargs = {})
triton_poi_fused_add_clamp_7 = async_compile.triton('triton_poi_fused_add_clamp_7', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 7, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/io/ciockrxdabkto2q62tpxuoga65qjcirohfznoyejp3ef5bqt347r.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   x_3 => add_33, clamp_max_9, clamp_min_6, clamp_min_9, convert_element_type_22, iota_3, mul_34, sub_21, sub_24
# Graph fragment:
#   %iota_3 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_3, torch.float32), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_22, 0.5), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_33, 0.5), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_34, 0.5), kwargs = {})
#   %clamp_min_6 : [num_users=4] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_21, 0.0), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_6, %convert_element_type_27), kwargs = {})
#   %clamp_min_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0.0), kwargs = {})
#   %clamp_max_9 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_9, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crzag5hzclzsdh3fgkngd24ns743sopglmayg4ttdijraspnbekc.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_3 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_8, _unsafe_index_9, add_39, add_40, add_41, add_42, add_43, add_44, add_45, mul_37, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, sub_25, sub_26, sub_27, sub_28, sub_30, sub_31, sub_33
# Graph fragment:
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_9, [None, None, %convert_element_type_23, %convert_element_type_25, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_9, [None, None, %convert_element_type_23, %convert_element_type_25, %clamp_max_8]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_9, [None, None, %convert_element_type_23, %clamp_max_7, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_9, [None, None, %convert_element_type_23, %clamp_max_7, %clamp_max_8]), kwargs = {})
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_9, [None, None, %clamp_max_6, %convert_element_type_25, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_9, [None, None, %clamp_max_6, %convert_element_type_25, %clamp_max_8]), kwargs = {})
#   %_unsafe_index_14 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_9, [None, None, %clamp_max_6, %clamp_max_7, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_9, [None, None, %clamp_max_6, %clamp_max_7, %clamp_max_8]), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %clamp_max_9), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_37), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %clamp_max_9), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_38), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %clamp_max_9), kwargs = {})
#   %add_41 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_39), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_15, %_unsafe_index_14), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %clamp_max_9), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_14, %mul_40), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_40, %add_39), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %clamp_max_10), kwargs = {})
#   %add_43 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %mul_41), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_42, %add_41), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %clamp_max_10), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_41, %mul_42), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_44, %add_43), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %clamp_max_11), kwargs = {})
#   %add_45 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %mul_43), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_9 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 16)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x3 = xindex // 4096
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp10 = tmp9 + tmp1
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tmp13 = tl.load(in_ptr3 + (tmp12 + 8*tmp8 + 64*tmp4 + 512*x3), None, eviction_policy='evict_last')
    tmp15 = tmp14 + tmp1
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tmp18 = tl.load(in_ptr3 + (tmp17 + 8*tmp8 + 64*tmp4 + 512*x3), None, eviction_policy='evict_last')
    tmp19 = tmp18 - tmp13
    tmp21 = tmp19 * tmp20
    tmp22 = tmp13 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tmp27 = tl.load(in_ptr3 + (tmp12 + 8*tmp8 + 64*tmp26 + 512*x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (tmp17 + 8*tmp8 + 64*tmp26 + 512*x3), None, eviction_policy='evict_last')
    tmp29 = tmp28 - tmp27
    tmp30 = tmp29 * tmp20
    tmp31 = tmp27 + tmp30
    tmp33 = tmp32 + tmp1
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tmp36 = tl.load(in_ptr3 + (tmp12 + 8*tmp35 + 64*tmp26 + 512*x3), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (tmp17 + 8*tmp35 + 64*tmp26 + 512*x3), None, eviction_policy='evict_last')
    tmp38 = tmp37 - tmp36
    tmp39 = tmp38 * tmp20
    tmp40 = tmp36 + tmp39
    tmp41 = tmp40 - tmp31
    tmp43 = tmp41 * tmp42
    tmp44 = tl.load(in_ptr3 + (tmp12 + 8*tmp35 + 64*tmp4 + 512*x3), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr3 + (tmp17 + 8*tmp35 + 64*tmp4 + 512*x3), None, eviction_policy='evict_last')
    tmp46 = tmp45 - tmp44
    tmp47 = tmp46 * tmp20
    tmp48 = tmp44 + tmp47
    tmp49 = tmp48 - tmp22
    tmp50 = tmp49 * tmp42
    tmp51 = tmp31 + tmp43
    tmp52 = tmp22 + tmp50
    tmp53 = tmp52 - tmp51
    tmp55 = tmp53 * tmp54
    tmp56 = tmp51 + tmp55
    tl.store(in_out_ptr0 + (x6), tmp56, None)
''', device_str='cuda')


# kernel path: inductor_cache/ce/ccekpomq3wu2mizlhvklth5o42auf5uqd54shr2rmhs7hry5fpmh.py
# Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_29 => add_47, mul_45, mul_46, sub_34
#   out_30 => relu_8
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_98), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_101), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %unsqueeze_104), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %unsqueeze_107), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x0), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/pp/cppzqyslcu4oxnzhbxllv77qw56uxsnwy7toozeycf5r5vhukbf7.py
# Topologically Sorted Source Nodes: [out_32, out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_32 => add_49, mul_48, mul_49, sub_35
#   out_33 => add_50
#   out_34 => relu_9
# Graph fragment:
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_110), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_113), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %unsqueeze_116), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_119), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %add_45), kwargs = {})
#   %relu_9 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_50,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr5 + (x0), None)
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x0), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/f7/cf7e2fphh7e63jd4hgjwgxn36eivchikjogyd2gbfptbcxt7zi25.py
# Topologically Sorted Source Nodes: [out_53, out_54, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_53 => add_64, mul_66, mul_67, sub_41
#   out_54 => add_65
#   out_55 => relu_15
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_182), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_185), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_66, %unsqueeze_188), kwargs = {})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_67, %unsqueeze_191), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_64, %relu_13), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_15, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp20 = tl.load(in_ptr5 + (x0), None)
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp24 = 0.0
    tmp25 = tmp23 <= tmp24
    tl.store(out_ptr0 + (x0), tmp23, None)
    tl.store(out_ptr1 + (x0), tmp25, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83 = args
    args.clear()
    assert_size_stride(primals_1, (2, 4, 1, 1, 1), (4, 1, 1, 1, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4, 4), (256, 64, 16, 4, 1))
    assert_size_stride(primals_3, (2, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_4, (2, ), (1, ))
    assert_size_stride(primals_5, (2, ), (1, ))
    assert_size_stride(primals_6, (2, ), (1, ))
    assert_size_stride(primals_7, (2, ), (1, ))
    assert_size_stride(primals_8, (2, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_9, (2, ), (1, ))
    assert_size_stride(primals_10, (2, ), (1, ))
    assert_size_stride(primals_11, (2, ), (1, ))
    assert_size_stride(primals_12, (2, ), (1, ))
    assert_size_stride(primals_13, (2, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_14, (2, ), (1, ))
    assert_size_stride(primals_15, (2, ), (1, ))
    assert_size_stride(primals_16, (2, ), (1, ))
    assert_size_stride(primals_17, (2, ), (1, ))
    assert_size_stride(primals_18, (2, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_19, (2, ), (1, ))
    assert_size_stride(primals_20, (2, ), (1, ))
    assert_size_stride(primals_21, (2, ), (1, ))
    assert_size_stride(primals_22, (2, ), (1, ))
    assert_size_stride(primals_23, (2, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_24, (2, ), (1, ))
    assert_size_stride(primals_25, (2, ), (1, ))
    assert_size_stride(primals_26, (2, ), (1, ))
    assert_size_stride(primals_27, (2, ), (1, ))
    assert_size_stride(primals_28, (2, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_29, (2, ), (1, ))
    assert_size_stride(primals_30, (2, ), (1, ))
    assert_size_stride(primals_31, (2, ), (1, ))
    assert_size_stride(primals_32, (2, ), (1, ))
    assert_size_stride(primals_33, (2, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_34, (2, ), (1, ))
    assert_size_stride(primals_35, (2, ), (1, ))
    assert_size_stride(primals_36, (2, ), (1, ))
    assert_size_stride(primals_37, (2, ), (1, ))
    assert_size_stride(primals_38, (2, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_39, (2, ), (1, ))
    assert_size_stride(primals_40, (2, ), (1, ))
    assert_size_stride(primals_41, (2, ), (1, ))
    assert_size_stride(primals_42, (2, ), (1, ))
    assert_size_stride(primals_43, (1, 2, 1, 1, 1), (2, 1, 1, 1, 1))
    assert_size_stride(primals_44, (1, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_45, (1, ), (1, ))
    assert_size_stride(primals_46, (1, ), (1, ))
    assert_size_stride(primals_47, (1, ), (1, ))
    assert_size_stride(primals_48, (1, ), (1, ))
    assert_size_stride(primals_49, (1, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_50, (1, ), (1, ))
    assert_size_stride(primals_51, (1, ), (1, ))
    assert_size_stride(primals_52, (1, ), (1, ))
    assert_size_stride(primals_53, (1, ), (1, ))
    assert_size_stride(primals_54, (1, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_55, (1, ), (1, ))
    assert_size_stride(primals_56, (1, ), (1, ))
    assert_size_stride(primals_57, (1, ), (1, ))
    assert_size_stride(primals_58, (1, ), (1, ))
    assert_size_stride(primals_59, (1, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_60, (1, ), (1, ))
    assert_size_stride(primals_61, (1, ), (1, ))
    assert_size_stride(primals_62, (1, ), (1, ))
    assert_size_stride(primals_63, (1, ), (1, ))
    assert_size_stride(primals_64, (1, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_65, (1, ), (1, ))
    assert_size_stride(primals_66, (1, ), (1, ))
    assert_size_stride(primals_67, (1, ), (1, ))
    assert_size_stride(primals_68, (1, ), (1, ))
    assert_size_stride(primals_69, (1, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_70, (1, ), (1, ))
    assert_size_stride(primals_71, (1, ), (1, ))
    assert_size_stride(primals_72, (1, ), (1, ))
    assert_size_stride(primals_73, (1, ), (1, ))
    assert_size_stride(primals_74, (1, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_75, (1, ), (1, ))
    assert_size_stride(primals_76, (1, ), (1, ))
    assert_size_stride(primals_77, (1, ), (1, ))
    assert_size_stride(primals_78, (1, ), (1, ))
    assert_size_stride(primals_79, (1, 1, 3, 3, 3), (27, 27, 9, 3, 1))
    assert_size_stride(primals_80, (1, ), (1, ))
    assert_size_stride(primals_81, (1, ), (1, ))
    assert_size_stride(primals_82, (1, ), (1, ))
    assert_size_stride(primals_83, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 2, 4, 4, 4), (128, 64, 16, 4, 1))
        buf1 = empty_strided_cuda((8, 1, 1), (1, 1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(buf1, 8, grid=grid(8), stream=stream0)
        buf2 = empty_strided_cuda((8, 1, 1), (1, 1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_1.run(buf2, 8, grid=grid(8), stream=stream0)
        buf3 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(buf3, 8, grid=grid(8), stream=stream0)
        buf4 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_1.run(buf4, 8, grid=grid(8), stream=stream0)
        buf5 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(buf5, 8, grid=grid(8), stream=stream0)
        buf6 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_1.run(buf6, 8, grid=grid(8), stream=stream0)
        buf7 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_2.run(buf7, 8, grid=grid(8), stream=stream0)
        buf10 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_2.run(buf10, 8, grid=grid(8), stream=stream0)
        buf13 = empty_strided_cuda((8, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_2.run(buf13, 8, grid=grid(8), stream=stream0)
        buf8 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        buf14 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_3.run(buf14, buf2, buf3, buf5, buf0, buf6, buf7, buf1, buf4, buf10, buf13, 4096, grid=grid(4096), stream=stream0)
        del buf0
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_3, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 2, 8, 8, 8), (1024, 512, 64, 8, 1))
        buf16 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf15, primals_4, primals_5, primals_6, primals_7, buf16, 4096, grid=grid(4096), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_8, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 2, 8, 8, 8), (1024, 512, 64, 8, 1))
        buf18 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf17, primals_9, primals_10, primals_11, primals_12, buf14, buf18, 4096, grid=grid(4096), stream=stream0)
        del primals_12
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_13, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 2, 8, 8, 8), (1024, 512, 64, 8, 1))
        buf20 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf19, primals_14, primals_15, primals_16, primals_17, buf20, 4096, grid=grid(4096), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_18, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 2, 8, 8, 8), (1024, 512, 64, 8, 1))
        buf22 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf21, primals_19, primals_20, primals_21, primals_22, buf18, buf22, 4096, grid=grid(4096), stream=stream0)
        del primals_22
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_23, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 2, 8, 8, 8), (1024, 512, 64, 8, 1))
        buf24 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf23, primals_24, primals_25, primals_26, primals_27, buf24, 4096, grid=grid(4096), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_28, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 2, 8, 8, 8), (1024, 512, 64, 8, 1))
        buf26 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_18, out_19, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf25, primals_29, primals_30, primals_31, primals_32, buf22, buf26, 4096, grid=grid(4096), stream=stream0)
        del primals_32
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_33, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 2, 8, 8, 8), (1024, 512, 64, 8, 1))
        buf28 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf27, primals_34, primals_35, primals_36, primals_37, buf28, 4096, grid=grid(4096), stream=stream0)
        del primals_37
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_38, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 2, 8, 8, 8), (1024, 512, 64, 8, 1))
        buf30 = empty_strided_cuda((4, 2, 8, 8, 8), (1024, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_25, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf29, primals_39, primals_40, primals_41, primals_42, buf26, buf30, 4096, grid=grid(4096), stream=stream0)
        del primals_42
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_43, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 1, 8, 8, 8), (512, 512, 64, 8, 1))
        buf32 = empty_strided_cuda((16, 1, 1), (1, 1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf32, 16, grid=grid(16), stream=stream0)
        buf33 = empty_strided_cuda((16, 1, 1), (1, 1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_7.run(buf33, 16, grid=grid(16), stream=stream0)
        buf34 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf34, 16, grid=grid(16), stream=stream0)
        buf35 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_7.run(buf35, 16, grid=grid(16), stream=stream0)
        buf36 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf36, 16, grid=grid(16), stream=stream0)
        buf37 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_7.run(buf37, 16, grid=grid(16), stream=stream0)
        buf38 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8.run(buf38, 16, grid=grid(16), stream=stream0)
        buf41 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8.run(buf41, 16, grid=grid(16), stream=stream0)
        buf44 = empty_strided_cuda((16, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8.run(buf44, 16, grid=grid(16), stream=stream0)
        buf39 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 16384, 256, 16, 1), torch.float32)
        buf45 = reinterpret_tensor(buf39, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_9.run(buf45, buf33, buf34, buf36, buf31, buf37, buf38, buf32, buf35, buf41, buf44, 16384, grid=grid(16384), stream=stream0)
        del buf31
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_44, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
        buf47 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf46, primals_45, primals_46, primals_47, primals_48, buf47, 16384, grid=grid(16384), stream=stream0)
        del primals_48
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_49, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
        buf49 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_32, out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf48, primals_50, primals_51, primals_52, primals_53, buf45, buf49, 16384, grid=grid(16384), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_54, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
        buf51 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_36, out_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf50, primals_55, primals_56, primals_57, primals_58, buf51, 16384, grid=grid(16384), stream=stream0)
        del primals_58
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_59, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
        buf53 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_39, out_40, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf52, primals_60, primals_61, primals_62, primals_63, buf49, buf53, 16384, grid=grid(16384), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_64, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
        buf55 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_43, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf54, primals_65, primals_66, primals_67, primals_68, buf55, 16384, grid=grid(16384), stream=stream0)
        del primals_68
        # Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_69, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
        buf57 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_46, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf56, primals_70, primals_71, primals_72, primals_73, buf53, buf57, 16384, grid=grid(16384), stream=stream0)
        del primals_73
        # Topologically Sorted Source Nodes: [out_49], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_74, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
        buf59 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_50, out_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf58, primals_75, primals_76, primals_77, primals_78, buf59, 16384, grid=grid(16384), stream=stream0)
        del primals_78
        # Topologically Sorted Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_79, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1))
        buf61 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.float32)
        buf62 = empty_strided_cuda((4, 1, 16, 16, 16), (4096, 4096, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_53, out_54, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_12.run(buf60, primals_80, primals_81, primals_82, primals_83, buf57, buf61, buf62, 16384, grid=grid(16384), stream=stream0)
        del primals_83
    return (buf61, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_54, primals_55, primals_56, primals_57, primals_59, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_67, primals_69, primals_70, primals_71, primals_72, primals_74, primals_75, primals_76, primals_77, primals_79, primals_80, primals_81, primals_82, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf10, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf41, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf62, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2, 4, 1, 1, 1), (4, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4, 4), (256, 64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((2, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((2, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((2, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((2, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1, 2, 1, 1, 1), (2, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1, 1, 3, 3, 3), (27, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
