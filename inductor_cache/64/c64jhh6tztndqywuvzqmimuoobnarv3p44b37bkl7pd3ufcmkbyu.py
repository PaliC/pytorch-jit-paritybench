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


# kernel path: inductor_cache/ep/cep7ej7hzso2tnojotha7k7ykoj4dawnil6tpgag46j2sycqq7j4.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.arange, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_2, add_4, add_5, clamp_max_2, clamp_max_3, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type_1, convert_element_type_2, convert_element_type_3, iota_1, mul_1, mul_2, mul_3, mul_4, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6
# Graph fragment:
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (299,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 0.013377926421404682), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0.0), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_1, torch.int64), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_3), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_4), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1072812
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 299) % 299)
    x0 = (xindex % 299)
    x2 = xindex // 89401
    x3 = (xindex % 89401)
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.013377926421404682
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 + tmp2
    tmp17 = tmp16 * tmp4
    tmp18 = tmp17 - tmp2
    tmp19 = triton_helpers.maximum(tmp18, tmp7)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tmp20 + tmp10
    tmp22 = triton_helpers.minimum(tmp21, tmp12)
    tmp23 = tl.load(in_ptr0 + (tmp22 + 4*tmp13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (tmp20 + 4*tmp13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20.to(tl.float32)
    tmp27 = tmp19 - tmp26
    tmp28 = triton_helpers.maximum(tmp27, tmp7)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = tl.load(in_ptr0 + (tmp20 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (tmp22 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp34 = tmp33 - tmp32
    tmp35 = tmp34 * tmp30
    tmp36 = tmp32 + tmp35
    tmp37 = tmp24 + tmp31
    tmp38 = tmp37 - tmp36
    tmp39 = tmp9.to(tl.float32)
    tmp40 = tmp8 - tmp39
    tmp41 = triton_helpers.maximum(tmp40, tmp7)
    tmp42 = triton_helpers.minimum(tmp41, tmp29)
    tmp43 = tmp38 * tmp42
    tl.store(out_ptr0 + (x3 + 89408*x2), tmp36, xmask)
    tl.store(in_out_ptr0 + (x3 + 89408*x2), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zs/czsjkplu4vuwjvbmqo4opacr5o5quwojzou6zsvommpfwnrzmygz.py
# Topologically Sorted Source Nodes: [x, mul, add, setitem, mul_1, add_1, setitem_1], Original ATen: [aten.add, aten.mul, aten.copy]
# Source node to ATen node mapping:
#   add => add_7
#   add_1 => add_8
#   mul => mul_5
#   mul_1 => mul_6
#   setitem => copy
#   setitem_1 => copy_1
#   x => add_6
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, 0.458), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, -0.030000000000000027), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_1, %add_7), kwargs = {})
#   %select_scatter_default : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%add_6, %copy, 1, 0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, 0.448), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, -0.08799999999999997), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_6, %add_8), kwargs = {})
#   %select_scatter_default_1 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %copy_1, 1, 1), kwargs = {})
triton_poi_fused_add_copy_mul_1 = async_compile.triton('triton_poi_fused_add_copy_mul_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1072812
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 89401) % 3)
    x0 = (xindex % 89401)
    x2 = xindex // 268203
    x3 = xindex // 89401
    x4 = xindex
    tmp5 = tl.load(in_ptr0 + (x0 + 268224*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0 + 268224*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (89408 + x0 + 268224*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (89408 + x0 + 268224*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (x0 + 89408*x3), xmask)
    tmp22 = tl.load(in_ptr1 + (x0 + 89408*x3), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = 0.458
    tmp9 = tmp7 * tmp8
    tmp10 = -0.030000000000000027
    tmp11 = tmp9 + tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp4, tmp11, tmp14)
    tmp16 = 0.448
    tmp17 = tmp15 * tmp16
    tmp18 = -0.08799999999999997
    tmp19 = tmp17 + tmp18
    tmp20 = tmp0 == tmp3
    tmp23 = tmp21 + tmp22
    tmp24 = tl.where(tmp20, tmp11, tmp23)
    tmp25 = tl.where(tmp2, tmp19, tmp24)
    tl.store(out_ptr0 + (x4), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/77/c772l3mg7zavmmaixhfdywjbaqm6raf6coeh7gu2x533uv3jzbet.py
# Topologically Sorted Source Nodes: [mul_2, add_2, setitem_2], Original ATen: [aten.mul, aten.add, aten.copy]
# Source node to ATen node mapping:
#   add_2 => add_9
#   mul_2 => mul_7
#   setitem_2 => copy_2
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, 0.45), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, -0.18799999999999994), kwargs = {})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_11, %add_9), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %copy_2, 1, 2), kwargs = {})
triton_poi_fused_add_copy_mul_2 = async_compile.triton('triton_poi_fused_add_copy_mul_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 131072}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy_mul_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 89401
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 3)
    x2 = xindex
    y1 = yindex // 3
    y3 = yindex
    tmp3 = tl.load(in_ptr0 + (178802 + x2 + 268203*y1), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (x2 + 89401*y3), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.45
    tmp5 = tmp3 * tmp4
    tmp6 = -0.18799999999999994
    tmp7 = tmp5 + tmp6
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tl.store(out_ptr0 + (y0 + 3*x2 + 268203*y1), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbrpqlb7bmgl3cd65bnnoiwhlrgdkq3eph4rt5glqerlt7yhirt2.py
# Topologically Sorted Source Nodes: [mul_2, add_2, setitem_2, x_2], Original ATen: [aten.mul, aten.add, aten.copy, aten.convolution]
# Source node to ATen node mapping:
#   add_2 => add_9
#   mul_2 => mul_7
#   setitem_2 => copy_2
#   x_2 => convolution
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, 0.45), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, -0.18799999999999994), kwargs = {})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_11, %add_9), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %copy_2, 1, 2), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%select_scatter_default_2, %arg1_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_copy_mul_3 = async_compile.triton('triton_poi_fused_add_convolution_copy_mul_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_copy_mul_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_copy_mul_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/co/ccowclot45epqsyopscp2tnmaemddgqboa7ytdcxlxcv3pvqtin3.py
# Topologically Sorted Source Nodes: [x_3, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => relu
#   x_3 => add_11, mul_10, mul_9, sub_7
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_5), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2841728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pq/cpqq62h3izidcdzmv3jj3mfb52lrnafasjukg6gl6ju4kyr6iwiz.py
# Topologically Sorted Source Nodes: [x_3, input_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_1 => relu
#   x_3 => add_11, mul_10, mul_9, sub_7
#   x_4 => convolution_1
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_5), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sm/csmct4g6sg55kv75up5iivqy4chhvqjyjthrsl6hzlk74diuonqy.py
# Topologically Sorted Source Nodes: [x_5, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => relu_1
#   x_5 => add_13, mul_12, mul_13, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_11), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_13), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2765952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gg/cggh62iazy3akid5oy5rcbo3hi7aadevaspawbv7uhnh6qeuehjt.py
# Topologically Sorted Source Nodes: [x_5, input_2, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_2 => relu_1
#   x_5 => add_13, mul_12, mul_13, sub_8
#   x_6 => convolution_2
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_11), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_13), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cj/ccjybvczgx5fzbzhidqax7wg2yoiprm5jq65t5krknz46jcnxc5e.py
# Topologically Sorted Source Nodes: [x_7, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_3 => relu_2
#   x_7 => add_15, mul_15, mul_16, sub_9
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_19), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %unsqueeze_21), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5531904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u7/cu7we7aygvlk5goiezb73lthjz2nfxwz5asdcxhxzu7o7pw3hw2m.py
# Topologically Sorted Source Nodes: [x_7, input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_3 => relu_2
#   input_4 => _low_memory_max_pool2d_with_offsets
#   x_7 => add_15, mul_15, mul_16, sub_9
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_19), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %unsqueeze_21), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1364224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 73)
    x2 = ((xindex // 4672) % 73)
    x3 = xindex // 341056
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (9408 + x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (9472 + x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (9536 + x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (18816 + x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (18880 + x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (18944 + x0 + 128*x1 + 18816*x2 + 1382976*x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z6/cz6hr4iattxic6ndahdcxek24yopqzvdxhstpdvrlwqn5giey7he.py
# Topologically Sorted Source Nodes: [x_9, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => relu_3
#   x_9 => add_17, mul_18, mul_19, sub_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_27), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_29), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 80)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/my/cmyacq7as2j6urtj43hz5b6pemghzf4yyeez2bcptq6ypwzrxvom.py
# Topologically Sorted Source Nodes: [x_9, input_5, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_5 => relu_3
#   x_10 => convolution_4
#   x_9 => add_17, mul_18, mul_19, sub_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_27), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_29), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 80)
    y1 = yindex // 80
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 80*x2 + 720*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6s/c6si6wwwzgsk75hcopp3w6ayoy2jofvjo5md24pxunups27uceg6.py
# Topologically Sorted Source Nodes: [x_11, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_6 => relu_4
#   x_11 => add_19, mul_21, mul_22, sub_11
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_35), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_37), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3871488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vk/cvk7yxupqkkrvglxbqf2jrizu5cpbjyxroe3rvwodmt6p5wj5jj7.py
# Topologically Sorted Source Nodes: [x_11, input_6, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_6 => relu_4
#   input_7 => _low_memory_max_pool2d_with_offsets_1
#   x_11 => add_19, mul_21, mul_22, sub_11
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_35), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_37), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 940800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 35)
    x2 = ((xindex // 6720) % 35)
    x3 = xindex // 235200
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp3 = tl.load(in_ptr0 + (384 + x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (13632 + x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp7 = tl.load(in_ptr0 + (13824 + x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp9 = tl.load(in_ptr0 + (14016 + x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp11 = tl.load(in_ptr0 + (27264 + x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp13 = tl.load(in_ptr0 + (27456 + x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp15 = tl.load(in_ptr0 + (27648 + x0 + 384*x1 + 27264*x2 + 967872*x3), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ny/cny7mkbe3hy5bwk27duoxcevbsvafhbtvglil3xdvhyjawkgys4t.py
# Topologically Sorted Source Nodes: [x_15, branch5x5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch5x5 => relu_6
#   x_15 => add_23, mul_27, mul_28, sub_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_51), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %unsqueeze_53), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 235200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lv/clv6rrqkn26q62tttt4vadqhzw56fp7hwmbpj56qoor4xj264nfc.py
# Topologically Sorted Source Nodes: [x_15, branch5x5, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch5x5 => relu_6
#   x_15 => add_23, mul_27, mul_28, sub_13
#   x_16 => convolution_7
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_51), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %unsqueeze_53), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg36_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 48)
    y1 = yindex // 48
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 48*x2 + 1200*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bk/cbkj7izpvw4vbgliwnn3vveabjmwbqg6mdbk5i3n7rnvcm6jdxko.py
# Topologically Sorted Source Nodes: [x_19, branch3x3dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl => relu_8
#   x_19 => add_27, mul_33, mul_34, sub_15
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_67), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %unsqueeze_69), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_27,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kh/ckhhzzxqzkbrhnngyzkesbatldqjz3bjhaj7mdm3plhimmti4edq.py
# Topologically Sorted Source Nodes: [x_19, branch3x3dbl, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch3x3dbl => relu_8
#   x_19 => add_27, mul_33, mul_34, sub_15
#   x_20 => convolution_9
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_67), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %unsqueeze_69), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_27,), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lj/cljrp43hzwxao7ef4tqd75r77vztkmm5xhvcc3wmfj3ikzogoeoz.py
# Topologically Sorted Source Nodes: [x_21, branch3x3dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_1 => relu_9
#   x_21 => add_29, mul_36, mul_37, sub_16
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_75), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %unsqueeze_77), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 470400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4v/c4vka7al725gvwktprnzfpnewcwr4uaqxtkbvjekqyloybzf5hxx.py
# Topologically Sorted Source Nodes: [x_21, branch3x3dbl_1, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch3x3dbl_1 => relu_9
#   x_21 => add_29, mul_36, mul_37, sub_16
#   x_22 => convolution_10
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_75), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %unsqueeze_77), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
#   %convolution_10 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg51_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4g/c4gk53ynx47ixsru4xr4tstdbm4puft4ygicnbh2nm2ufaxftulb.py
# Topologically Sorted Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_2, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_20 = async_compile.triton('triton_poi_fused_avg_pool2d_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 940800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 6720) % 35)
    x1 = ((xindex // 192) % 35)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6912) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-6720) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-6528) + x6), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-192) + x6), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (192 + x6), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (6528 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (6720 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (6912 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36)))*((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36))) + ((-1)*x1*((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36)))) + ((-1)*x2*((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36)))) + ((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36))) + ((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fx/cfxuyffzhfgc7c6fp5egtdgcvkysmckswhsffowdunmm3v5rq5ef.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_8 => cat
# Graph fragment:
#   %cat : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_5, %relu_7, %relu_10, %relu_11], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 128, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (64*x1 + ((-64) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 224, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (96*x1 + ((-128) + x0)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 256, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (32*x1 + ((-224) + x0)), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x2), tmp98, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5k/c5kqaa4fy63g6h4mmdtxgdbebrhxxqa65rumeavfdegaspjlkclq.py
# Topologically Sorted Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_2 => avg_pool2d_1
# Graph fragment:
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_22 = async_compile.triton('triton_poi_fused_avg_pool2d_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 8960) % 35)
    x1 = ((xindex // 256) % 35)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9216) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-8960) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8704) + x6), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-256) + x6), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (256 + x6), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (8704 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (8960 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (9216 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36)))*((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36))) + ((-1)*x1*((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36)))) + ((-1)*x2*((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36)))) + ((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36))) + ((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nl/cnlyop7gz5x2pk2lyaazm7k6osknuhnpmnb2yzi426m32smkagt2.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_9 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %relu_14, %relu_17, %relu_18], 1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1411200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 288)
    x1 = xindex // 288
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 128, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (64*x1 + ((-64) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-64) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 224, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (96*x1 + ((-128) + x0)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-128) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 288, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (64*x1 + ((-224) + x0)), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-224) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x2), tmp98, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7j/c7jrxstvclh2aupzclm4a372lgwwoy7zstra55zyopnvebcd45sd.py
# Topologically Sorted Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_4 => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_1, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_24 = async_compile.triton('triton_poi_fused_avg_pool2d_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1411200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 10080) % 35)
    x1 = ((xindex // 288) % 35)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-10368) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-10080) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-9792) + x6), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-288) + x6), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (288 + x6), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (9792 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (10080 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (10368 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36)))*((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36))) + ((-1)*x1*((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36)))) + ((-1)*x2*((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36)))) + ((36) * ((36) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (36))) + ((36) * ((36) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (36)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x6/cx6od2mbre4bbbfupuvlbpss5x4vh6jsqesbtsop5gcoudpyt2nt.py
# Topologically Sorted Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   branch_pool_6 => _low_memory_max_pool2d_with_offsets_2
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_2, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_25 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1156
    xnumel = 288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 17)
    y1 = ((yindex // 17) % 17)
    y2 = yindex // 289
    y4 = (yindex % 289)
    tmp0 = tl.load(in_ptr0 + (x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (288 + x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (576 + x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (10080 + x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (10368 + x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (10656 + x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (20160 + x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (20448 + x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (20736 + x3 + 576*y0 + 20160*y1 + 352800*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y4 + 289*x3 + 221952*y2), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ug/cugfvrcu6dooqnmwelplumobdjmbyicjnlgh6eqw6haz3rubz2w5.py
# Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_54 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg131_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_26 = async_compile.triton('triton_poi_fused_convolution_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 110592
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 288)
    y1 = yindex // 288
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 288*x2 + 2592*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/3c/c3ciadn2m6pzd3rkjzrpe4urbkeskccohb26vqlid43brtosqiiq.py
# Topologically Sorted Source Nodes: [x_55, branch3x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3 => relu_26
#   x_55 => add_63, mul_87, mul_88, sub_33
# Graph fragment:
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_211), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %unsqueeze_213), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (y0 + 384*x2 + 110976*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 289*y0 + 221952*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/lv/clvllqrnich3xgxxejmw4zptj3pusn2wbk36gad6relh4xbkebkz.py
# Topologically Sorted Source Nodes: [x_61, branch3x3dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_11 => relu_29
#   x_61 => add_69, mul_96, mul_97, sub_36
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_235), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %unsqueeze_237), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %unsqueeze_239), kwargs = {})
#   %relu_29 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_69,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 27744*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 289*y0 + 221952*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4rnneett6okjia5wtfz2xfhswrwg2hoxr24ssxgxzupp5utevyy.py
# Topologically Sorted Source Nodes: [x_62, x_64, x_70, branch_pool_7], Original ATen: [aten.convolution, aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_7 => avg_pool2d_3
#   x_62 => convolution_30
#   x_64 => convolution_31
#   x_70 => convolution_34
# Graph fragment:
#   %convolution_30 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_34 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_3, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_29 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_29(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y5 = yindex
    y0 = (yindex % 768)
    y1 = yindex // 768
    x4 = xindex // 17
    x3 = (xindex % 17)
    tmp0 = tl.load(in_ptr0 + (x2 + 289*y5), xmask, eviction_policy='evict_last')
    tmp1 = (-1) + x4
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 17, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (-1) + x3
    tmp8 = tmp7 >= tmp2
    tmp9 = tmp7 < tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp6 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-18) + x2 + 289*y5), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = x3
    tmp14 = tmp13 >= tmp2
    tmp15 = tmp13 < tmp4
    tmp16 = tmp14 & tmp15
    tmp17 = tmp6 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-17) + x2 + 289*y5), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 + tmp12
    tmp20 = 1 + x3
    tmp21 = tmp20 >= tmp2
    tmp22 = tmp20 < tmp4
    tmp23 = tmp21 & tmp22
    tmp24 = tmp6 & tmp23
    tmp25 = tl.load(in_ptr0 + ((-16) + x2 + 289*y5), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 + tmp19
    tmp27 = x4
    tmp28 = tmp27 >= tmp2
    tmp29 = tmp27 < tmp4
    tmp30 = tmp28 & tmp29
    tmp31 = tmp30 & tmp10
    tmp32 = tl.load(in_ptr0 + ((-1) + x2 + 289*y5), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp26
    tmp34 = tmp30 & tmp16
    tmp35 = tl.load(in_ptr0 + (x2 + 289*y5), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = tmp30 & tmp23
    tmp38 = tl.load(in_ptr0 + (1 + x2 + 289*y5), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38 + tmp36
    tmp40 = 1 + x4
    tmp41 = tmp40 >= tmp2
    tmp42 = tmp40 < tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp10
    tmp45 = tl.load(in_ptr0 + (16 + x2 + 289*y5), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp39
    tmp47 = tmp43 & tmp16
    tmp48 = tl.load(in_ptr0 + (17 + x2 + 289*y5), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 + tmp46
    tmp50 = tmp43 & tmp23
    tmp51 = tl.load(in_ptr0 + (18 + x2 + 289*y5), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 + tmp49
    tmp53 = 1 + ((-1)*x3) + ((-1)*x4) + x3*x4 + ((18) * ((18) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (18)))*((18) * ((18) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (18))) + ((-1)*x3*((18) * ((18) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (18)))) + ((-1)*x4*((18) * ((18) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (18)))) + ((18) * ((18) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (18))) + ((18) * ((18) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (18)))
    tmp54 = tmp52 / tmp53
    tl.store(out_ptr0 + (y0 + 768*x2 + 221952*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 768*x2 + 221952*y1), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + 768*x2 + 221952*y1), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + 768*x2 + 221952*y1), tmp54, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/we/cwerjs33i6l7lcgbyt7h67fcg5qbpr7h5asyc7ykl7om244sej7y.py
# Topologically Sorted Source Nodes: [x_65, branch7x7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7 => relu_31
#   x_65 => add_73, mul_102, mul_103, sub_38
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_251), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %unsqueeze_253), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %unsqueeze_255), kwargs = {})
#   %relu_31 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_73,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xy/cxymguakjhes57uifjzbup2qofsu3yqfftenqujejf35vvmsykuh.py
# Topologically Sorted Source Nodes: [x_65, branch7x7, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch7x7 => relu_31
#   x_65 => add_73, mul_102, mul_103, sub_38
#   x_66 => convolution_32
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_251), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %unsqueeze_253), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %unsqueeze_255), kwargs = {})
#   %relu_31 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_73,), kwargs = {})
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_31, %arg161_1, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 896*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4p/c4pmfyq4tuh47nxz6qndnfmei4oitmgjqtt4zv4pwyci7exyinle.py
# Topologically Sorted Source Nodes: [x_67, branch7x7_1, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch7x7_1 => relu_32
#   x_67 => add_75, mul_105, mul_106, sub_39
#   x_68 => convolution_33
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_257), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_259), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_105, %unsqueeze_261), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %unsqueeze_263), kwargs = {})
#   %relu_32 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_75,), kwargs = {})
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_32, %arg166_1, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 896*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ix/cixmgmkcqz3krgidd6bnozkagonb5xsj3or6oo3cvzdevomaowht.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_12 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_30, %relu_33, %relu_38, %relu_39], 1), kwargs = {})
triton_poi_fused_cat_33 = async_compile.triton('triton_poi_fused_cat_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 887808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 768)
    x1 = xindex // 768
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (192*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 384, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (192*x1 + ((-192) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-192) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-192) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-192) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-192) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 576, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (192*x1 + ((-384) + x0)), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-384) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-384) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-384) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-384) + x0), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 768, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (192*x1 + ((-576) + x0)), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-576) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-576) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-576) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-576) + x0), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x2), tmp98, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dl/cdlmziq72yozkcmjqs2m7purmmprsyjrjihgttyc35dcmi5cjcjh.py
# Topologically Sorted Source Nodes: [x_85, branch7x7_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7_3 => relu_41
#   x_85 => add_93, mul_132, mul_133, sub_48
# Graph fragment:
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_331), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_132, %unsqueeze_333), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_133, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_93,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 184960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u7/cu7xqxpwx3hiisdw6tsp6ywjbzv3qlsuclrhhznx3dygpuaftfmk.py
# Topologically Sorted Source Nodes: [x_85, branch7x7_3, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch7x7_3 => relu_41
#   x_85 => add_93, mul_132, mul_133, sub_48
#   x_86 => convolution_42
# Graph fragment:
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_331), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_132, %unsqueeze_333), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_133, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_93,), kwargs = {})
#   %convolution_42 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %arg211_1, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 160)
    y1 = yindex // 160
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1120*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k3/ck3gjsux7xrrmf7pgs4alvtegh7guocyioizqqfj2pb3zpbtvgcg.py
# Topologically Sorted Source Nodes: [x_87, branch7x7_4, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch7x7_4 => relu_42
#   x_87 => add_95, mul_135, mul_136, sub_49
#   x_88 => convolution_43
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_339), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_135, %unsqueeze_341), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, %unsqueeze_343), kwargs = {})
#   %relu_42 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_95,), kwargs = {})
#   %convolution_43 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_42, %arg216_1, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30720
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 160)
    y1 = yindex // 160
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1120*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zr/czr7w2eewoonffbj4hyytqak6oiksgk4khvnb3cyoyhqwlgh52e3.py
# Topologically Sorted Source Nodes: [branch_pool_9], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_9 => avg_pool2d_4
# Graph fragment:
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_4, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_37 = async_compile.triton('triton_poi_fused_avg_pool2d_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 887808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 13056) % 17)
    x1 = ((xindex // 768) % 17)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 17, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-13824) + x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-13056) + x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-12288) + x6), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-768) + x6), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (768 + x6), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (12288 + x6), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (13056 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (13824 + x6), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*x1) + ((-1)*x2) + x1*x2 + ((18) * ((18) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (18)))*((18) * ((18) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (18))) + ((-1)*x1*((18) * ((18) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (18)))) + ((-1)*x2*((18) * ((18) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (18)))) + ((18) * ((18) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (18))) + ((18) * ((18) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (18)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ms/cms3imf4hfbcberbszfq463ypixcitboo5lud22wrrqq53rs3jz7.py
# Topologically Sorted Source Nodes: [x_125, branch7x7_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7_9 => relu_61
#   x_125 => add_133, mul_192, mul_193, sub_68
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_489), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_491), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_192, %unsqueeze_493), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_193, %unsqueeze_495), kwargs = {})
#   %relu_61 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_133,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 221952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f7/cf7l24ggdskrr2fkfhgujnn5vywkppfjzk47kz3ctslyxdbwwmdt.py
# Topologically Sorted Source Nodes: [x_125, branch7x7_9, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch7x7_9 => relu_61
#   x_125 => add_133, mul_192, mul_193, sub_68
#   x_126 => convolution_62
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_489), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_491), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_192, %unsqueeze_493), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_193, %unsqueeze_495), kwargs = {})
#   %relu_61 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_133,), kwargs = {})
#   %convolution_62 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_61, %arg311_1, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1344*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d7/cd7kcyuvrd4srcb4hmw2shklo2f6a6ph7volzg3jsuvtleomlful.py
# Topologically Sorted Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   branch_pool_15 => _low_memory_max_pool2d_with_offsets_3
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_40 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_40(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 8)
    y1 = ((yindex // 8) % 8)
    y2 = yindex // 64
    y4 = (yindex % 64)
    tmp0 = tl.load(in_ptr0 + (x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (768 + x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1536 + x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13056 + x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (13824 + x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (14592 + x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26112 + x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (26880 + x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (27648 + x3 + 1536*y0 + 26112*y1 + 221952*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y4 + 64*x3 + 81920*y2), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/d4/cd4iqzuma4oy7dkeileox5sf2m6jjdayj2cvpujufekpr7m7cgak.py
# Topologically Sorted Source Nodes: [x_143, branch3x3_1, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch3x3_1 => relu_70
#   x_143 => add_151, mul_219, mul_220, sub_77
#   x_144 => convolution_71
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_70, %unsqueeze_561), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_563), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_219, %unsqueeze_565), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_220, %unsqueeze_567), kwargs = {})
#   %relu_70 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_151,), kwargs = {})
#   %convolution_71 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_70, %arg356_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 61440
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/py/cpyqh5jvz4rf2k3ddrsk3ha77sqwbywhn65ejef7hl3ohd7hp2su.py
# Topologically Sorted Source Nodes: [x_151, branch7x7x3_2, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch7x7x3_2 => relu_74
#   x_151 => add_159, mul_231, mul_232, sub_81
#   x_152 => convolution_75
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_74, %unsqueeze_593), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_595), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_231, %unsqueeze_597), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_232, %unsqueeze_599), kwargs = {})
#   %relu_74 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_159,), kwargs = {})
#   %convolution_75 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_74, %arg376_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rg/crg3wqeemgbspfk2q6epbt6fr5j2gae6j4ijmos7m3pkhefxpqv7.py
# Topologically Sorted Source Nodes: [x_145, branch3x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3_2 => relu_71
#   x_145 => add_153, mul_222, mul_223, sub_78
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_569), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_571), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_222, %unsqueeze_573), kwargs = {})
#   %add_153 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_223, %unsqueeze_575), kwargs = {})
#   %relu_71 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_153,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (y0 + 320*x2 + 20480*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 64*y0 + 81920*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/is/cisw2p5mwqh2tlxxbizm3xpf7ff2kwrxmh2j3adxfky7ocorotve.py
# Topologically Sorted Source Nodes: [x_153, branch7x7x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch7x7x3_3 => relu_75
#   x_153 => add_161, mul_234, mul_235, sub_82
# Graph fragment:
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_75, %unsqueeze_601), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %unsqueeze_603), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_234, %unsqueeze_605), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_235, %unsqueeze_607), kwargs = {})
#   %relu_75 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_161,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 12288*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 64*y0 + 81920*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zw/czw7iv4ljdeqkuku7cl245up7khedmuivcfbr3khfdnyh6czkequ.py
# Topologically Sorted Source Nodes: [x_154, x_156, x_162, branch_pool_16], Original ATen: [aten.convolution, aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_16 => avg_pool2d_7
#   x_154 => convolution_76
#   x_156 => convolution_77
#   x_162 => convolution_80
# Graph fragment:
#   %convolution_76 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_8, %arg381_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_77 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_8, %arg386_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_8, %arg401_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %avg_pool2d_7 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_8, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_45 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_45(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y5 = yindex
    y0 = (yindex % 1280)
    y1 = yindex // 1280
    x4 = xindex // 8
    x3 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y5), xmask, eviction_policy='evict_last')
    tmp1 = (-1) + x4
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 8, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (-1) + x3
    tmp8 = tmp7 >= tmp2
    tmp9 = tmp7 < tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp6 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-9) + x2 + 64*y5), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = x3
    tmp14 = tmp13 >= tmp2
    tmp15 = tmp13 < tmp4
    tmp16 = tmp14 & tmp15
    tmp17 = tmp6 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-8) + x2 + 64*y5), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 + tmp12
    tmp20 = 1 + x3
    tmp21 = tmp20 >= tmp2
    tmp22 = tmp20 < tmp4
    tmp23 = tmp21 & tmp22
    tmp24 = tmp6 & tmp23
    tmp25 = tl.load(in_ptr0 + ((-7) + x2 + 64*y5), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 + tmp19
    tmp27 = x4
    tmp28 = tmp27 >= tmp2
    tmp29 = tmp27 < tmp4
    tmp30 = tmp28 & tmp29
    tmp31 = tmp30 & tmp10
    tmp32 = tl.load(in_ptr0 + ((-1) + x2 + 64*y5), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp26
    tmp34 = tmp30 & tmp16
    tmp35 = tl.load(in_ptr0 + (x2 + 64*y5), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = tmp30 & tmp23
    tmp38 = tl.load(in_ptr0 + (1 + x2 + 64*y5), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38 + tmp36
    tmp40 = 1 + x4
    tmp41 = tmp40 >= tmp2
    tmp42 = tmp40 < tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp10
    tmp45 = tl.load(in_ptr0 + (7 + x2 + 64*y5), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp39
    tmp47 = tmp43 & tmp16
    tmp48 = tl.load(in_ptr0 + (8 + x2 + 64*y5), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 + tmp46
    tmp50 = tmp43 & tmp23
    tmp51 = tl.load(in_ptr0 + (9 + x2 + 64*y5), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 + tmp49
    tmp53 = 1 + ((-1)*x3) + ((-1)*x4) + x3*x4 + ((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9)))*((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9))) + ((-1)*x3*((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9)))) + ((-1)*x4*((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9)))) + ((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9))) + ((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9)))
    tmp54 = tmp52 / tmp53
    tl.store(out_ptr0 + (y0 + 1280*x2 + 81920*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 1280*x2 + 81920*y1), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + 1280*x2 + 81920*y1), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + 1280*x2 + 81920*y1), tmp54, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/56/c56ygkjc46p7eghl3vtrb2mdkoeqt66ie6ed3zxjidt7tzj5e35u.py
# Topologically Sorted Source Nodes: [x_157, branch3x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3_3 => relu_77
#   x_157 => add_165, mul_240, mul_241, sub_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_617), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_619), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_240, %unsqueeze_621), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_241, %unsqueeze_623), kwargs = {})
#   %relu_77 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_165,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ln/clnkeytogb225k3rf7rzj2ihitavbh3x7dvyhcjte5kzjnoweuez.py
# Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_158 => convolution_78
# Graph fragment:
#   %convolution_78 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_77, %arg391_1, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_47 = async_compile.triton('triton_poi_fused_convolution_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
    xnumel = 3
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    tmp0 = tl.load(in_ptr0 + (x2 + 3*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 384*x2 + 1152*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/r5/cr5pgj2wzzlzfo2twdp7lglwmgleol6bthke2ais2yls7etetymb.py
# Topologically Sorted Source Nodes: [branch3x3_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   branch3x3_4 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_78, %relu_79], 1), kwargs = {})
triton_poi_fused_cat_48 = async_compile.triton('triton_poi_fused_cat_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 768)
    x0 = (xindex % 64)
    x2 = xindex // 49152
    x3 = (xindex % 49152)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (384*x0 + 24576*x2 + (x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 768, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + (384*x0 + 24576*x2 + ((-384) + x1)), tmp25, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-384) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr7 + ((-384) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp32 = 0.001
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr8 + ((-384) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr9 + ((-384) + x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp25, tmp45, tmp46)
    tmp48 = tl.where(tmp4, tmp24, tmp47)
    tl.store(out_ptr0 + (x3 + 131072*x2), tmp48, None)
''', device_str='cuda')


# kernel path: inductor_cache/xu/cxuay4hzinywjjcqyxrezb2lklhnb6rlcvnhwezqm5qgd5di65yc.py
# Topologically Sorted Source Nodes: [x_163, branch3x3dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch3x3dbl_12 => relu_80
#   x_163 => add_171, mul_249, mul_250, sub_87
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_641), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_643), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_249, %unsqueeze_645), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_250, %unsqueeze_647), kwargs = {})
#   %relu_80 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_171,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_49', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 114688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 448)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/cj/ccj4hzxme5s4men2hfyu65mxhkssw3gurgszcrn734kpqdkipcad.py
# Topologically Sorted Source Nodes: [x_163, branch3x3dbl_12, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   branch3x3dbl_12 => relu_80
#   x_163 => add_171, mul_249, mul_250, sub_87
#   x_164 => convolution_81
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_641), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_643), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_249, %unsqueeze_645), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_250, %unsqueeze_647), kwargs = {})
#   %relu_80 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_171,), kwargs = {})
#   %convolution_81 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_80, %arg406_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_50', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_50(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 172032
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 448)
    y1 = yindex // 448
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 448*x2 + 4032*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ob/cobgyrpsuxcj2bi5qntean4jm4d2pf2dbhhvsyx2kxwhmz2nvi76.py
# Topologically Sorted Source Nodes: [x_155, branch1x1_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch1x1_7 => relu_76
#   x_155 => add_163, mul_237, mul_238, sub_83
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_609), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_611), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_237, %unsqueeze_613), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %unsqueeze_615), kwargs = {})
#   %relu_76 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_163,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (y0 + 320*x2 + 20480*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 64*y0 + 131072*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/e5/ce5iy4qjzw4w7a5nikysewd64sdibyfmj2y5qotsngdn27wka47a.py
# Topologically Sorted Source Nodes: [x_171, branch_pool_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   branch_pool_17 => relu_84
#   x_171 => add_179, mul_261, mul_262, sub_91
# Graph fragment:
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_673), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_675), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_261, %unsqueeze_677), kwargs = {})
#   %add_179 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_262, %unsqueeze_679), kwargs = {})
#   %relu_84 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_179,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (y0 + 192*x2 + 12288*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 64*y0 + 131072*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zq/czqpaohorn336szp7eaqcmk6opgxjio5vqavqxu5udfeidarmx32.py
# Topologically Sorted Source Nodes: [x_172, x_174, x_180, branch_pool_18], Original ATen: [aten.convolution, aten.avg_pool2d]
# Source node to ATen node mapping:
#   branch_pool_18 => avg_pool2d_8
#   x_172 => convolution_85
#   x_174 => convolution_86
#   x_180 => convolution_89
# Graph fragment:
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg426_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_86 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg431_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_89 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg446_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %avg_pool2d_8 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%cat_11, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_53 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_53(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y5 = yindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    x4 = xindex // 8
    x3 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y5), xmask, eviction_policy='evict_last')
    tmp1 = (-1) + x4
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 8, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (-1) + x3
    tmp8 = tmp7 >= tmp2
    tmp9 = tmp7 < tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp6 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-9) + x2 + 64*y5), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = x3
    tmp14 = tmp13 >= tmp2
    tmp15 = tmp13 < tmp4
    tmp16 = tmp14 & tmp15
    tmp17 = tmp6 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-8) + x2 + 64*y5), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 + tmp12
    tmp20 = 1 + x3
    tmp21 = tmp20 >= tmp2
    tmp22 = tmp20 < tmp4
    tmp23 = tmp21 & tmp22
    tmp24 = tmp6 & tmp23
    tmp25 = tl.load(in_ptr0 + ((-7) + x2 + 64*y5), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 + tmp19
    tmp27 = x4
    tmp28 = tmp27 >= tmp2
    tmp29 = tmp27 < tmp4
    tmp30 = tmp28 & tmp29
    tmp31 = tmp30 & tmp10
    tmp32 = tl.load(in_ptr0 + ((-1) + x2 + 64*y5), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp26
    tmp34 = tmp30 & tmp16
    tmp35 = tl.load(in_ptr0 + (x2 + 64*y5), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = tmp30 & tmp23
    tmp38 = tl.load(in_ptr0 + (1 + x2 + 64*y5), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38 + tmp36
    tmp40 = 1 + x4
    tmp41 = tmp40 >= tmp2
    tmp42 = tmp40 < tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp10
    tmp45 = tl.load(in_ptr0 + (7 + x2 + 64*y5), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp39
    tmp47 = tmp43 & tmp16
    tmp48 = tl.load(in_ptr0 + (8 + x2 + 64*y5), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 + tmp46
    tmp50 = tmp43 & tmp23
    tmp51 = tl.load(in_ptr0 + (9 + x2 + 64*y5), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 + tmp49
    tmp53 = 1 + ((-1)*x3) + ((-1)*x4) + x3*x4 + ((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9)))*((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9))) + ((-1)*x3*((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9)))) + ((-1)*x4*((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9)))) + ((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9))) + ((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9)))
    tmp54 = tmp52 / tmp53
    tl.store(out_ptr0 + (y0 + 2048*x2 + 131072*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 2048*x2 + 131072*y1), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + 2048*x2 + 131072*y1), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + 2048*x2 + 131072*y1), tmp54, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fx/cfx53gfmhlehkljgtrbwik4g6cxxemgnucn5w4weztljyz7drtxp.py
# Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   input_19 => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%cat_14, [-1, -2], True), kwargs = {})
triton_per_fused_mean_54 = async_compile.triton('triton_per_fused_mean_54', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_54(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 64.0
    tmp5 = tmp3 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg1_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg17_1, (80, ), (1, ))
    assert_size_stride(arg18_1, (80, ), (1, ))
    assert_size_stride(arg19_1, (80, ), (1, ))
    assert_size_stride(arg20_1, (80, ), (1, ))
    assert_size_stride(arg21_1, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(arg22_1, (192, ), (1, ))
    assert_size_stride(arg23_1, (192, ), (1, ))
    assert_size_stride(arg24_1, (192, ), (1, ))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg32_1, (48, ), (1, ))
    assert_size_stride(arg33_1, (48, ), (1, ))
    assert_size_stride(arg34_1, (48, ), (1, ))
    assert_size_stride(arg35_1, (48, ), (1, ))
    assert_size_stride(arg36_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg47_1, (96, ), (1, ))
    assert_size_stride(arg48_1, (96, ), (1, ))
    assert_size_stride(arg49_1, (96, ), (1, ))
    assert_size_stride(arg50_1, (96, ), (1, ))
    assert_size_stride(arg51_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg52_1, (96, ), (1, ))
    assert_size_stride(arg53_1, (96, ), (1, ))
    assert_size_stride(arg54_1, (96, ), (1, ))
    assert_size_stride(arg55_1, (96, ), (1, ))
    assert_size_stride(arg56_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg57_1, (32, ), (1, ))
    assert_size_stride(arg58_1, (32, ), (1, ))
    assert_size_stride(arg59_1, (32, ), (1, ))
    assert_size_stride(arg60_1, (32, ), (1, ))
    assert_size_stride(arg61_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (64, ), (1, ))
    assert_size_stride(arg64_1, (64, ), (1, ))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg67_1, (48, ), (1, ))
    assert_size_stride(arg68_1, (48, ), (1, ))
    assert_size_stride(arg69_1, (48, ), (1, ))
    assert_size_stride(arg70_1, (48, ), (1, ))
    assert_size_stride(arg71_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg72_1, (64, ), (1, ))
    assert_size_stride(arg73_1, (64, ), (1, ))
    assert_size_stride(arg74_1, (64, ), (1, ))
    assert_size_stride(arg75_1, (64, ), (1, ))
    assert_size_stride(arg76_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg77_1, (64, ), (1, ))
    assert_size_stride(arg78_1, (64, ), (1, ))
    assert_size_stride(arg79_1, (64, ), (1, ))
    assert_size_stride(arg80_1, (64, ), (1, ))
    assert_size_stride(arg81_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg82_1, (96, ), (1, ))
    assert_size_stride(arg83_1, (96, ), (1, ))
    assert_size_stride(arg84_1, (96, ), (1, ))
    assert_size_stride(arg85_1, (96, ), (1, ))
    assert_size_stride(arg86_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg87_1, (96, ), (1, ))
    assert_size_stride(arg88_1, (96, ), (1, ))
    assert_size_stride(arg89_1, (96, ), (1, ))
    assert_size_stride(arg90_1, (96, ), (1, ))
    assert_size_stride(arg91_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg92_1, (64, ), (1, ))
    assert_size_stride(arg93_1, (64, ), (1, ))
    assert_size_stride(arg94_1, (64, ), (1, ))
    assert_size_stride(arg95_1, (64, ), (1, ))
    assert_size_stride(arg96_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg97_1, (64, ), (1, ))
    assert_size_stride(arg98_1, (64, ), (1, ))
    assert_size_stride(arg99_1, (64, ), (1, ))
    assert_size_stride(arg100_1, (64, ), (1, ))
    assert_size_stride(arg101_1, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg102_1, (48, ), (1, ))
    assert_size_stride(arg103_1, (48, ), (1, ))
    assert_size_stride(arg104_1, (48, ), (1, ))
    assert_size_stride(arg105_1, (48, ), (1, ))
    assert_size_stride(arg106_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg107_1, (64, ), (1, ))
    assert_size_stride(arg108_1, (64, ), (1, ))
    assert_size_stride(arg109_1, (64, ), (1, ))
    assert_size_stride(arg110_1, (64, ), (1, ))
    assert_size_stride(arg111_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg112_1, (64, ), (1, ))
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (64, ), (1, ))
    assert_size_stride(arg116_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg117_1, (96, ), (1, ))
    assert_size_stride(arg118_1, (96, ), (1, ))
    assert_size_stride(arg119_1, (96, ), (1, ))
    assert_size_stride(arg120_1, (96, ), (1, ))
    assert_size_stride(arg121_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg122_1, (96, ), (1, ))
    assert_size_stride(arg123_1, (96, ), (1, ))
    assert_size_stride(arg124_1, (96, ), (1, ))
    assert_size_stride(arg125_1, (96, ), (1, ))
    assert_size_stride(arg126_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg127_1, (64, ), (1, ))
    assert_size_stride(arg128_1, (64, ), (1, ))
    assert_size_stride(arg129_1, (64, ), (1, ))
    assert_size_stride(arg130_1, (64, ), (1, ))
    assert_size_stride(arg131_1, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (384, ), (1, ))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg137_1, (64, ), (1, ))
    assert_size_stride(arg138_1, (64, ), (1, ))
    assert_size_stride(arg139_1, (64, ), (1, ))
    assert_size_stride(arg140_1, (64, ), (1, ))
    assert_size_stride(arg141_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg142_1, (96, ), (1, ))
    assert_size_stride(arg143_1, (96, ), (1, ))
    assert_size_stride(arg144_1, (96, ), (1, ))
    assert_size_stride(arg145_1, (96, ), (1, ))
    assert_size_stride(arg146_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg147_1, (96, ), (1, ))
    assert_size_stride(arg148_1, (96, ), (1, ))
    assert_size_stride(arg149_1, (96, ), (1, ))
    assert_size_stride(arg150_1, (96, ), (1, ))
    assert_size_stride(arg151_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg152_1, (192, ), (1, ))
    assert_size_stride(arg153_1, (192, ), (1, ))
    assert_size_stride(arg154_1, (192, ), (1, ))
    assert_size_stride(arg155_1, (192, ), (1, ))
    assert_size_stride(arg156_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (128, ), (1, ))
    assert_size_stride(arg159_1, (128, ), (1, ))
    assert_size_stride(arg160_1, (128, ), (1, ))
    assert_size_stride(arg161_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (128, ), (1, ))
    assert_size_stride(arg165_1, (128, ), (1, ))
    assert_size_stride(arg166_1, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg167_1, (192, ), (1, ))
    assert_size_stride(arg168_1, (192, ), (1, ))
    assert_size_stride(arg169_1, (192, ), (1, ))
    assert_size_stride(arg170_1, (192, ), (1, ))
    assert_size_stride(arg171_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (128, ), (1, ))
    assert_size_stride(arg179_1, (128, ), (1, ))
    assert_size_stride(arg180_1, (128, ), (1, ))
    assert_size_stride(arg181_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (128, ), (1, ))
    assert_size_stride(arg186_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (128, ), (1, ))
    assert_size_stride(arg189_1, (128, ), (1, ))
    assert_size_stride(arg190_1, (128, ), (1, ))
    assert_size_stride(arg191_1, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg192_1, (192, ), (1, ))
    assert_size_stride(arg193_1, (192, ), (1, ))
    assert_size_stride(arg194_1, (192, ), (1, ))
    assert_size_stride(arg195_1, (192, ), (1, ))
    assert_size_stride(arg196_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg197_1, (192, ), (1, ))
    assert_size_stride(arg198_1, (192, ), (1, ))
    assert_size_stride(arg199_1, (192, ), (1, ))
    assert_size_stride(arg200_1, (192, ), (1, ))
    assert_size_stride(arg201_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg202_1, (192, ), (1, ))
    assert_size_stride(arg203_1, (192, ), (1, ))
    assert_size_stride(arg204_1, (192, ), (1, ))
    assert_size_stride(arg205_1, (192, ), (1, ))
    assert_size_stride(arg206_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg207_1, (160, ), (1, ))
    assert_size_stride(arg208_1, (160, ), (1, ))
    assert_size_stride(arg209_1, (160, ), (1, ))
    assert_size_stride(arg210_1, (160, ), (1, ))
    assert_size_stride(arg211_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg212_1, (160, ), (1, ))
    assert_size_stride(arg213_1, (160, ), (1, ))
    assert_size_stride(arg214_1, (160, ), (1, ))
    assert_size_stride(arg215_1, (160, ), (1, ))
    assert_size_stride(arg216_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg217_1, (192, ), (1, ))
    assert_size_stride(arg218_1, (192, ), (1, ))
    assert_size_stride(arg219_1, (192, ), (1, ))
    assert_size_stride(arg220_1, (192, ), (1, ))
    assert_size_stride(arg221_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg222_1, (160, ), (1, ))
    assert_size_stride(arg223_1, (160, ), (1, ))
    assert_size_stride(arg224_1, (160, ), (1, ))
    assert_size_stride(arg225_1, (160, ), (1, ))
    assert_size_stride(arg226_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg227_1, (160, ), (1, ))
    assert_size_stride(arg228_1, (160, ), (1, ))
    assert_size_stride(arg229_1, (160, ), (1, ))
    assert_size_stride(arg230_1, (160, ), (1, ))
    assert_size_stride(arg231_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg232_1, (160, ), (1, ))
    assert_size_stride(arg233_1, (160, ), (1, ))
    assert_size_stride(arg234_1, (160, ), (1, ))
    assert_size_stride(arg235_1, (160, ), (1, ))
    assert_size_stride(arg236_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg237_1, (160, ), (1, ))
    assert_size_stride(arg238_1, (160, ), (1, ))
    assert_size_stride(arg239_1, (160, ), (1, ))
    assert_size_stride(arg240_1, (160, ), (1, ))
    assert_size_stride(arg241_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg242_1, (192, ), (1, ))
    assert_size_stride(arg243_1, (192, ), (1, ))
    assert_size_stride(arg244_1, (192, ), (1, ))
    assert_size_stride(arg245_1, (192, ), (1, ))
    assert_size_stride(arg246_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg247_1, (192, ), (1, ))
    assert_size_stride(arg248_1, (192, ), (1, ))
    assert_size_stride(arg249_1, (192, ), (1, ))
    assert_size_stride(arg250_1, (192, ), (1, ))
    assert_size_stride(arg251_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg252_1, (192, ), (1, ))
    assert_size_stride(arg253_1, (192, ), (1, ))
    assert_size_stride(arg254_1, (192, ), (1, ))
    assert_size_stride(arg255_1, (192, ), (1, ))
    assert_size_stride(arg256_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg257_1, (160, ), (1, ))
    assert_size_stride(arg258_1, (160, ), (1, ))
    assert_size_stride(arg259_1, (160, ), (1, ))
    assert_size_stride(arg260_1, (160, ), (1, ))
    assert_size_stride(arg261_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg262_1, (160, ), (1, ))
    assert_size_stride(arg263_1, (160, ), (1, ))
    assert_size_stride(arg264_1, (160, ), (1, ))
    assert_size_stride(arg265_1, (160, ), (1, ))
    assert_size_stride(arg266_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg267_1, (192, ), (1, ))
    assert_size_stride(arg268_1, (192, ), (1, ))
    assert_size_stride(arg269_1, (192, ), (1, ))
    assert_size_stride(arg270_1, (192, ), (1, ))
    assert_size_stride(arg271_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg272_1, (160, ), (1, ))
    assert_size_stride(arg273_1, (160, ), (1, ))
    assert_size_stride(arg274_1, (160, ), (1, ))
    assert_size_stride(arg275_1, (160, ), (1, ))
    assert_size_stride(arg276_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg277_1, (160, ), (1, ))
    assert_size_stride(arg278_1, (160, ), (1, ))
    assert_size_stride(arg279_1, (160, ), (1, ))
    assert_size_stride(arg280_1, (160, ), (1, ))
    assert_size_stride(arg281_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg282_1, (160, ), (1, ))
    assert_size_stride(arg283_1, (160, ), (1, ))
    assert_size_stride(arg284_1, (160, ), (1, ))
    assert_size_stride(arg285_1, (160, ), (1, ))
    assert_size_stride(arg286_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg287_1, (160, ), (1, ))
    assert_size_stride(arg288_1, (160, ), (1, ))
    assert_size_stride(arg289_1, (160, ), (1, ))
    assert_size_stride(arg290_1, (160, ), (1, ))
    assert_size_stride(arg291_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg292_1, (192, ), (1, ))
    assert_size_stride(arg293_1, (192, ), (1, ))
    assert_size_stride(arg294_1, (192, ), (1, ))
    assert_size_stride(arg295_1, (192, ), (1, ))
    assert_size_stride(arg296_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg297_1, (192, ), (1, ))
    assert_size_stride(arg298_1, (192, ), (1, ))
    assert_size_stride(arg299_1, (192, ), (1, ))
    assert_size_stride(arg300_1, (192, ), (1, ))
    assert_size_stride(arg301_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg302_1, (192, ), (1, ))
    assert_size_stride(arg303_1, (192, ), (1, ))
    assert_size_stride(arg304_1, (192, ), (1, ))
    assert_size_stride(arg305_1, (192, ), (1, ))
    assert_size_stride(arg306_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg307_1, (192, ), (1, ))
    assert_size_stride(arg308_1, (192, ), (1, ))
    assert_size_stride(arg309_1, (192, ), (1, ))
    assert_size_stride(arg310_1, (192, ), (1, ))
    assert_size_stride(arg311_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg312_1, (192, ), (1, ))
    assert_size_stride(arg313_1, (192, ), (1, ))
    assert_size_stride(arg314_1, (192, ), (1, ))
    assert_size_stride(arg315_1, (192, ), (1, ))
    assert_size_stride(arg316_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg317_1, (192, ), (1, ))
    assert_size_stride(arg318_1, (192, ), (1, ))
    assert_size_stride(arg319_1, (192, ), (1, ))
    assert_size_stride(arg320_1, (192, ), (1, ))
    assert_size_stride(arg321_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg322_1, (192, ), (1, ))
    assert_size_stride(arg323_1, (192, ), (1, ))
    assert_size_stride(arg324_1, (192, ), (1, ))
    assert_size_stride(arg325_1, (192, ), (1, ))
    assert_size_stride(arg326_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg327_1, (192, ), (1, ))
    assert_size_stride(arg328_1, (192, ), (1, ))
    assert_size_stride(arg329_1, (192, ), (1, ))
    assert_size_stride(arg330_1, (192, ), (1, ))
    assert_size_stride(arg331_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg332_1, (192, ), (1, ))
    assert_size_stride(arg333_1, (192, ), (1, ))
    assert_size_stride(arg334_1, (192, ), (1, ))
    assert_size_stride(arg335_1, (192, ), (1, ))
    assert_size_stride(arg336_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg337_1, (192, ), (1, ))
    assert_size_stride(arg338_1, (192, ), (1, ))
    assert_size_stride(arg339_1, (192, ), (1, ))
    assert_size_stride(arg340_1, (192, ), (1, ))
    assert_size_stride(arg341_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg342_1, (192, ), (1, ))
    assert_size_stride(arg343_1, (192, ), (1, ))
    assert_size_stride(arg344_1, (192, ), (1, ))
    assert_size_stride(arg345_1, (192, ), (1, ))
    assert_size_stride(arg346_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg347_1, (192, ), (1, ))
    assert_size_stride(arg348_1, (192, ), (1, ))
    assert_size_stride(arg349_1, (192, ), (1, ))
    assert_size_stride(arg350_1, (192, ), (1, ))
    assert_size_stride(arg351_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg352_1, (192, ), (1, ))
    assert_size_stride(arg353_1, (192, ), (1, ))
    assert_size_stride(arg354_1, (192, ), (1, ))
    assert_size_stride(arg355_1, (192, ), (1, ))
    assert_size_stride(arg356_1, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg357_1, (320, ), (1, ))
    assert_size_stride(arg358_1, (320, ), (1, ))
    assert_size_stride(arg359_1, (320, ), (1, ))
    assert_size_stride(arg360_1, (320, ), (1, ))
    assert_size_stride(arg361_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg362_1, (192, ), (1, ))
    assert_size_stride(arg363_1, (192, ), (1, ))
    assert_size_stride(arg364_1, (192, ), (1, ))
    assert_size_stride(arg365_1, (192, ), (1, ))
    assert_size_stride(arg366_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg367_1, (192, ), (1, ))
    assert_size_stride(arg368_1, (192, ), (1, ))
    assert_size_stride(arg369_1, (192, ), (1, ))
    assert_size_stride(arg370_1, (192, ), (1, ))
    assert_size_stride(arg371_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg372_1, (192, ), (1, ))
    assert_size_stride(arg373_1, (192, ), (1, ))
    assert_size_stride(arg374_1, (192, ), (1, ))
    assert_size_stride(arg375_1, (192, ), (1, ))
    assert_size_stride(arg376_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg377_1, (192, ), (1, ))
    assert_size_stride(arg378_1, (192, ), (1, ))
    assert_size_stride(arg379_1, (192, ), (1, ))
    assert_size_stride(arg380_1, (192, ), (1, ))
    assert_size_stride(arg381_1, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg382_1, (320, ), (1, ))
    assert_size_stride(arg383_1, (320, ), (1, ))
    assert_size_stride(arg384_1, (320, ), (1, ))
    assert_size_stride(arg385_1, (320, ), (1, ))
    assert_size_stride(arg386_1, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg387_1, (384, ), (1, ))
    assert_size_stride(arg388_1, (384, ), (1, ))
    assert_size_stride(arg389_1, (384, ), (1, ))
    assert_size_stride(arg390_1, (384, ), (1, ))
    assert_size_stride(arg391_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg392_1, (384, ), (1, ))
    assert_size_stride(arg393_1, (384, ), (1, ))
    assert_size_stride(arg394_1, (384, ), (1, ))
    assert_size_stride(arg395_1, (384, ), (1, ))
    assert_size_stride(arg396_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg397_1, (384, ), (1, ))
    assert_size_stride(arg398_1, (384, ), (1, ))
    assert_size_stride(arg399_1, (384, ), (1, ))
    assert_size_stride(arg400_1, (384, ), (1, ))
    assert_size_stride(arg401_1, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg402_1, (448, ), (1, ))
    assert_size_stride(arg403_1, (448, ), (1, ))
    assert_size_stride(arg404_1, (448, ), (1, ))
    assert_size_stride(arg405_1, (448, ), (1, ))
    assert_size_stride(arg406_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg407_1, (384, ), (1, ))
    assert_size_stride(arg408_1, (384, ), (1, ))
    assert_size_stride(arg409_1, (384, ), (1, ))
    assert_size_stride(arg410_1, (384, ), (1, ))
    assert_size_stride(arg411_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg412_1, (384, ), (1, ))
    assert_size_stride(arg413_1, (384, ), (1, ))
    assert_size_stride(arg414_1, (384, ), (1, ))
    assert_size_stride(arg415_1, (384, ), (1, ))
    assert_size_stride(arg416_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg417_1, (384, ), (1, ))
    assert_size_stride(arg418_1, (384, ), (1, ))
    assert_size_stride(arg419_1, (384, ), (1, ))
    assert_size_stride(arg420_1, (384, ), (1, ))
    assert_size_stride(arg421_1, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg422_1, (192, ), (1, ))
    assert_size_stride(arg423_1, (192, ), (1, ))
    assert_size_stride(arg424_1, (192, ), (1, ))
    assert_size_stride(arg425_1, (192, ), (1, ))
    assert_size_stride(arg426_1, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg427_1, (320, ), (1, ))
    assert_size_stride(arg428_1, (320, ), (1, ))
    assert_size_stride(arg429_1, (320, ), (1, ))
    assert_size_stride(arg430_1, (320, ), (1, ))
    assert_size_stride(arg431_1, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg432_1, (384, ), (1, ))
    assert_size_stride(arg433_1, (384, ), (1, ))
    assert_size_stride(arg434_1, (384, ), (1, ))
    assert_size_stride(arg435_1, (384, ), (1, ))
    assert_size_stride(arg436_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg437_1, (384, ), (1, ))
    assert_size_stride(arg438_1, (384, ), (1, ))
    assert_size_stride(arg439_1, (384, ), (1, ))
    assert_size_stride(arg440_1, (384, ), (1, ))
    assert_size_stride(arg441_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg442_1, (384, ), (1, ))
    assert_size_stride(arg443_1, (384, ), (1, ))
    assert_size_stride(arg444_1, (384, ), (1, ))
    assert_size_stride(arg445_1, (384, ), (1, ))
    assert_size_stride(arg446_1, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg447_1, (448, ), (1, ))
    assert_size_stride(arg448_1, (448, ), (1, ))
    assert_size_stride(arg449_1, (448, ), (1, ))
    assert_size_stride(arg450_1, (448, ), (1, ))
    assert_size_stride(arg451_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg452_1, (384, ), (1, ))
    assert_size_stride(arg453_1, (384, ), (1, ))
    assert_size_stride(arg454_1, (384, ), (1, ))
    assert_size_stride(arg455_1, (384, ), (1, ))
    assert_size_stride(arg456_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg457_1, (384, ), (1, ))
    assert_size_stride(arg458_1, (384, ), (1, ))
    assert_size_stride(arg459_1, (384, ), (1, ))
    assert_size_stride(arg460_1, (384, ), (1, ))
    assert_size_stride(arg461_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg462_1, (384, ), (1, ))
    assert_size_stride(arg463_1, (384, ), (1, ))
    assert_size_stride(arg464_1, (384, ), (1, ))
    assert_size_stride(arg465_1, (384, ), (1, ))
    assert_size_stride(arg466_1, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg467_1, (192, ), (1, ))
    assert_size_stride(arg468_1, (192, ), (1, ))
    assert_size_stride(arg469_1, (192, ), (1, ))
    assert_size_stride(arg470_1, (192, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 299, 299), (268224, 89408, 299, 1), torch.float32)
        buf1 = empty_strided_cuda((4, 3, 299, 299), (268224, 89408, 299, 1), torch.float32)
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.arange, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0.run(buf2, arg0_1, buf1, 1072812, grid=grid(1072812), stream=stream0)
        del arg0_1
        buf3 = empty_strided_cuda((4, 3, 299, 299), (268203, 89401, 299, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, mul, add, setitem, mul_1, add_1, setitem_1], Original ATen: [aten.add, aten.mul, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy_mul_1.run(buf1, buf2, buf3, 1072812, grid=grid(1072812), stream=stream0)
        del buf1
        del buf2
        buf4 = empty_strided_cuda((4, 3, 299, 299), (268203, 1, 897, 3), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, add_2, setitem_2], Original ATen: [aten.mul, aten.add, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy_mul_2.run(buf3, buf4, 12, 89401, grid=grid(12, 89401), stream=stream0)
        del buf3
        buf5 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, add_2, setitem_2, x_2], Original ATen: [aten.mul, aten.add, aten.copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_copy_mul_3.run(arg1_1, buf5, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [mul_2, add_2, setitem_2, x_2], Original ATen: [aten.mul, aten.add, aten.copy, aten.convolution]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 32, 149, 149), (710432, 1, 4768, 32))
        del buf4
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_3, input_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf7, arg2_1, arg3_1, arg4_1, arg5_1, 2841728, grid=grid(2841728), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf8 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, input_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg6_1, buf8, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [x_3, input_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 32, 147, 147), (691488, 1, 4704, 32))
        del buf7
        del buf8
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_5, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf10, arg7_1, arg8_1, arg9_1, arg10_1, 2765952, grid=grid(2765952), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf11 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, input_2, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(arg11_1, buf11, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [x_5, input_2, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 147, 147), (1382976, 1, 9408, 64))
        del buf10
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_7, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf13, arg12_1, arg13_1, arg14_1, arg15_1, 5531904, grid=grid(5531904), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        buf14 = empty_strided_cuda((4, 64, 73, 73), (341056, 1, 4672, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_9.run(buf13, buf14, 1364224, grid=grid(1364224), stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 80, 73, 73), (426320, 1, 5840, 80))
        del arg16_1
        del buf14
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_9, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf16, arg17_1, arg18_1, arg19_1, arg20_1, 1705280, grid=grid(1705280), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        buf17 = empty_strided_cuda((192, 80, 3, 3), (720, 1, 240, 80), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, input_5, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(arg21_1, buf17, 15360, 9, grid=grid(15360, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [x_9, input_5, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf18 = extern_kernels.convolution(buf16, buf17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 192, 71, 71), (967872, 1, 13632, 192))
        del buf16
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_11, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf19, arg22_1, arg23_1, arg24_1, arg25_1, 3871488, grid=grid(3871488), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        buf20 = empty_strided_cuda((4, 192, 35, 35), (235200, 1, 6720, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, input_6, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_13.run(buf19, buf20, 940800, grid=grid(940800), stream=stream0)
        del buf19
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg26_1
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf20, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 48, 35, 35), (58800, 1, 1680, 48))
        del arg31_1
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_15, branch5x5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf23, arg32_1, arg33_1, arg34_1, arg35_1, 235200, grid=grid(235200), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        buf24 = empty_strided_cuda((64, 48, 5, 5), (1200, 1, 240, 48), torch.float32)
        # Topologically Sorted Source Nodes: [x_15, branch5x5, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(arg36_1, buf24, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del arg36_1
        # Topologically Sorted Source Nodes: [x_15, branch5x5, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf25 = extern_kernels.convolution(buf23, buf24, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del buf23
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf20, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg41_1
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_19, branch3x3dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf27, arg42_1, arg43_1, arg44_1, arg45_1, 313600, grid=grid(313600), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf28 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, branch3x3dbl, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(arg46_1, buf28, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg46_1
        # Topologically Sorted Source Nodes: [x_19, branch3x3dbl, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 96, 35, 35), (117600, 1, 3360, 96))
        del buf27
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_21, branch3x3dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf30, arg47_1, arg48_1, arg49_1, arg50_1, 470400, grid=grid(470400), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        buf31 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_21, branch3x3dbl_1, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg51_1, buf31, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg51_1
        # Topologically Sorted Source Nodes: [x_21, branch3x3dbl_1, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf32 = extern_kernels.convolution(buf30, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 96, 35, 35), (117600, 1, 3360, 96))
        del buf30
        buf33 = empty_strided_cuda((4, 192, 35, 35), (235200, 1, 6720, 192), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_20.run(buf20, buf33, 940800, grid=grid(940800), stream=stream0)
        del buf20
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 32, 35, 35), (39200, 1, 1120, 32))
        del arg56_1
        del buf33
        buf35 = empty_strided_cuda((4, 256, 35, 35), (313600, 1, 8960, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf21, arg27_1, arg28_1, arg29_1, arg30_1, buf25, arg37_1, arg38_1, arg39_1, arg40_1, buf32, arg52_1, arg53_1, arg54_1, arg55_1, buf34, arg57_1, arg58_1, arg59_1, arg60_1, buf35, 1254400, grid=grid(1254400), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf21
        del buf25
        del buf32
        del buf34
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg61_1
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf35, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 48, 35, 35), (58800, 1, 1680, 48))
        del arg66_1
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_29, branch5x5_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf38, arg67_1, arg68_1, arg69_1, arg70_1, 235200, grid=grid(235200), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf39 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_29, branch5x5_2, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(arg71_1, buf39, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del arg71_1
        # Topologically Sorted Source Nodes: [x_29, branch5x5_2, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del buf38
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf35, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg76_1
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_33, branch3x3dbl_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf42, arg77_1, arg78_1, arg79_1, arg80_1, 313600, grid=grid(313600), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        buf43 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_33, branch3x3dbl_3, x_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(arg81_1, buf43, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [x_33, branch3x3dbl_3, x_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf44 = extern_kernels.convolution(buf42, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 96, 35, 35), (117600, 1, 3360, 96))
        del buf42
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_35, branch3x3dbl_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf45, arg82_1, arg83_1, arg84_1, arg85_1, 470400, grid=grid(470400), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf46 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_35, branch3x3dbl_4, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg86_1, buf46, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg86_1
        # Topologically Sorted Source Nodes: [x_35, branch3x3dbl_4, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf47 = extern_kernels.convolution(buf45, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 96, 35, 35), (117600, 1, 3360, 96))
        del buf45
        buf48 = empty_strided_cuda((4, 256, 35, 35), (313600, 1, 8960, 256), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_22.run(buf35, buf48, 1254400, grid=grid(1254400), stream=stream0)
        del buf35
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg91_1
        del buf48
        buf50 = empty_strided_cuda((4, 288, 35, 35), (352800, 1, 10080, 288), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf36, arg62_1, arg63_1, arg64_1, arg65_1, buf40, arg72_1, arg73_1, arg74_1, arg75_1, buf47, arg87_1, arg88_1, arg89_1, arg90_1, buf49, arg92_1, arg93_1, arg94_1, arg95_1, buf50, 1411200, grid=grid(1411200), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf36
        del buf40
        del buf47
        del buf49
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg96_1
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf50, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 48, 35, 35), (58800, 1, 1680, 48))
        del arg101_1
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_43, branch5x5_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf53, arg102_1, arg103_1, arg104_1, arg105_1, 235200, grid=grid(235200), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        buf54 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_43, branch5x5_4, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(arg106_1, buf54, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del arg106_1
        # Topologically Sorted Source Nodes: [x_43, branch5x5_4, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del buf53
        del buf54
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf50, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg111_1
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_47, branch3x3dbl_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf57, arg112_1, arg113_1, arg114_1, arg115_1, 313600, grid=grid(313600), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        buf58 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_47, branch3x3dbl_6, x_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(arg116_1, buf58, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg116_1
        # Topologically Sorted Source Nodes: [x_47, branch3x3dbl_6, x_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 96, 35, 35), (117600, 1, 3360, 96))
        del buf57
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_49, branch3x3dbl_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf60, arg117_1, arg118_1, arg119_1, arg120_1, 470400, grid=grid(470400), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        buf61 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_49, branch3x3dbl_7, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg121_1, buf61, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg121_1
        # Topologically Sorted Source Nodes: [x_49, branch3x3dbl_7, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 96, 35, 35), (117600, 1, 3360, 96))
        del buf60
        buf63 = empty_strided_cuda((4, 288, 35, 35), (352800, 1, 10080, 288), torch.float32)
        # Topologically Sorted Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_24.run(buf50, buf63, 1411200, grid=grid(1411200), stream=stream0)
        del buf50
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg126_1
        buf65 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf51, arg97_1, arg98_1, arg99_1, arg100_1, buf55, arg107_1, arg108_1, arg109_1, arg110_1, buf62, arg122_1, arg123_1, arg124_1, arg125_1, buf64, arg127_1, arg128_1, arg129_1, arg130_1, buf65, 1411200, grid=grid(1411200), stream=stream0)
        del arg100_1
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf51
        del buf55
        del buf62
        del buf64
        buf78 = empty_strided_cuda((4, 768, 17, 17), (221952, 289, 17, 1), torch.float32)
        buf66 = reinterpret_tensor(buf78, (4, 288, 17, 17), (221952, 289, 17, 1), 138720)  # alias
        # Topologically Sorted Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_25.run(buf65, buf66, 1156, 288, grid=grid(1156, 288), stream=stream0)
        buf67 = empty_strided_cuda((384, 288, 3, 3), (2592, 1, 864, 288), torch.float32)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_26.run(arg131_1, buf67, 110592, 9, grid=grid(110592, 9), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf65, buf67, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 384, 17, 17), (110976, 1, 6528, 384))
        del buf67
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf65, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 64, 35, 35), (78400, 1, 2240, 64))
        del arg136_1
        del buf65
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_57, branch3x3dbl_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf70, arg137_1, arg138_1, arg139_1, arg140_1, 313600, grid=grid(313600), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        buf71 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_57, branch3x3dbl_9, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(arg141_1, buf71, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg141_1
        # Topologically Sorted Source Nodes: [x_57, branch3x3dbl_9, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf72 = extern_kernels.convolution(buf70, buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 96, 35, 35), (117600, 1, 3360, 96))
        del buf70
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_59, branch3x3dbl_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf73, arg142_1, arg143_1, arg144_1, arg145_1, 470400, grid=grid(470400), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        buf74 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_59, branch3x3dbl_10, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg146_1, buf74, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [x_59, branch3x3dbl_10, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf75 = extern_kernels.convolution(buf73, buf74, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 96, 17, 17), (27744, 1, 1632, 96))
        del buf73
        del buf74
        buf76 = reinterpret_tensor(buf78, (4, 384, 17, 17), (221952, 289, 17, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_55, branch3x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf68, arg132_1, arg133_1, arg134_1, arg135_1, buf76, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del buf68
        buf77 = reinterpret_tensor(buf78, (4, 96, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        # Topologically Sorted Source Nodes: [x_61, branch3x3dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf75, arg147_1, arg148_1, arg149_1, arg150_1, buf77, 384, 289, grid=grid(384, 289), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        del buf75
        buf79 = empty_strided_cuda((4, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        buf81 = empty_strided_cuda((4, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        buf89 = empty_strided_cuda((4, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        buf103 = empty_strided_cuda((4, 768, 17, 17), (221952, 1, 13056, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_62, x_64, x_70, branch_pool_7], Original ATen: [aten.convolution, aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_convolution_29.run(buf78, buf79, buf81, buf89, buf103, 3072, 289, grid=grid(3072, 289), stream=stream0)
        del buf66
        del buf76
        del buf77
        del buf78
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg151_1
        del buf79
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 128, 17, 17), (36992, 1, 2176, 128))
        del arg156_1
        del buf81
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_65, branch7x7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf83, arg157_1, arg158_1, arg159_1, arg160_1, 147968, grid=grid(147968), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        buf84 = empty_strided_cuda((128, 128, 1, 7), (896, 1, 896, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_65, branch7x7, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.run(arg161_1, buf84, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del arg161_1
        # Topologically Sorted Source Nodes: [x_65, branch7x7, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf85 = extern_kernels.convolution(buf83, buf84, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 128, 17, 17), (36992, 1, 2176, 128))
        del buf83
        buf86 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_67, branch7x7_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf86, arg162_1, arg163_1, arg164_1, arg165_1, 147968, grid=grid(147968), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        buf87 = empty_strided_cuda((192, 128, 7, 1), (896, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_67, branch7x7_1, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32.run(arg166_1, buf87, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del arg166_1
        # Topologically Sorted Source Nodes: [x_67, branch7x7_1, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf88 = extern_kernels.convolution(buf86, buf87, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf86
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 128, 17, 17), (36992, 1, 2176, 128))
        del arg171_1
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_71, branch7x7dbl], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf91, arg172_1, arg173_1, arg174_1, arg175_1, 147968, grid=grid(147968), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        buf92 = reinterpret_tensor(buf84, (128, 128, 7, 1), (896, 1, 128, 128), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_71, branch7x7dbl, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.run(arg176_1, buf92, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [x_71, branch7x7dbl, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf93 = extern_kernels.convolution(buf91, buf92, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 128, 17, 17), (36992, 1, 2176, 128))
        del buf91
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_73, branch7x7dbl_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf94, arg177_1, arg178_1, arg179_1, arg180_1, 147968, grid=grid(147968), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        buf95 = reinterpret_tensor(buf92, (128, 128, 1, 7), (896, 1, 896, 128), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_73, branch7x7dbl_1, x_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.run(arg181_1, buf95, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del arg181_1
        # Topologically Sorted Source Nodes: [x_73, branch7x7dbl_1, x_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf96 = extern_kernels.convolution(buf94, buf95, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 128, 17, 17), (36992, 1, 2176, 128))
        del buf94
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_75, branch7x7dbl_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf97, arg182_1, arg183_1, arg184_1, arg185_1, 147968, grid=grid(147968), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        buf98 = reinterpret_tensor(buf95, (128, 128, 7, 1), (896, 1, 128, 128), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [x_75, branch7x7dbl_2, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31.run(arg186_1, buf98, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del arg186_1
        # Topologically Sorted Source Nodes: [x_75, branch7x7dbl_2, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf99 = extern_kernels.convolution(buf97, buf98, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 128, 17, 17), (36992, 1, 2176, 128))
        del buf97
        del buf98
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_77, branch7x7dbl_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf100, arg187_1, arg188_1, arg189_1, arg190_1, 147968, grid=grid(147968), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        buf101 = reinterpret_tensor(buf87, (192, 128, 1, 7), (896, 1, 896, 128), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_77, branch7x7dbl_3, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32.run(arg191_1, buf101, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del arg191_1
        # Topologically Sorted Source Nodes: [x_77, branch7x7dbl_3, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf102 = extern_kernels.convolution(buf100, buf101, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf100
        del buf101
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg196_1
        buf105 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf80, arg152_1, arg153_1, arg154_1, arg155_1, buf88, arg167_1, arg168_1, arg169_1, arg170_1, buf102, arg192_1, arg193_1, arg194_1, arg195_1, buf104, arg197_1, arg198_1, arg199_1, arg200_1, buf105, 887808, grid=grid(887808), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf102
        del buf104
        del buf80
        del buf88
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg201_1
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf105, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del arg206_1
        buf108 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_85, branch7x7_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf108, arg207_1, arg208_1, arg209_1, arg210_1, 184960, grid=grid(184960), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        buf109 = empty_strided_cuda((160, 160, 1, 7), (1120, 1, 1120, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_85, branch7x7_3, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35.run(arg211_1, buf109, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg211_1
        # Topologically Sorted Source Nodes: [x_85, branch7x7_3, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf110 = extern_kernels.convolution(buf108, buf109, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del buf108
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_87, branch7x7_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf111, arg212_1, arg213_1, arg214_1, arg215_1, 184960, grid=grid(184960), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        buf112 = empty_strided_cuda((192, 160, 7, 1), (1120, 1, 160, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_87, branch7x7_4, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(arg216_1, buf112, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del arg216_1
        # Topologically Sorted Source Nodes: [x_87, branch7x7_4, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf113 = extern_kernels.convolution(buf111, buf112, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf111
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf105, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del arg221_1
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_91, branch7x7dbl_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf115, arg222_1, arg223_1, arg224_1, arg225_1, 184960, grid=grid(184960), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf116 = reinterpret_tensor(buf109, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_91, branch7x7dbl_5, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35.run(arg226_1, buf116, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg226_1
        # Topologically Sorted Source Nodes: [x_91, branch7x7dbl_5, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf117 = extern_kernels.convolution(buf115, buf116, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del buf115
        buf118 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_93, branch7x7dbl_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf118, arg227_1, arg228_1, arg229_1, arg230_1, 184960, grid=grid(184960), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        buf119 = reinterpret_tensor(buf116, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_93, branch7x7dbl_6, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35.run(arg231_1, buf119, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg231_1
        # Topologically Sorted Source Nodes: [x_93, branch7x7dbl_6, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf120 = extern_kernels.convolution(buf118, buf119, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del buf118
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_95, branch7x7dbl_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf121, arg232_1, arg233_1, arg234_1, arg235_1, 184960, grid=grid(184960), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        buf122 = reinterpret_tensor(buf119, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_95, branch7x7dbl_7, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35.run(arg236_1, buf122, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg236_1
        # Topologically Sorted Source Nodes: [x_95, branch7x7dbl_7, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf123 = extern_kernels.convolution(buf121, buf122, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del buf121
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_97, branch7x7dbl_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf124, arg237_1, arg238_1, arg239_1, arg240_1, 184960, grid=grid(184960), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        buf125 = reinterpret_tensor(buf112, (192, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_97, branch7x7dbl_8, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(arg241_1, buf125, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del arg241_1
        # Topologically Sorted Source Nodes: [x_97, branch7x7dbl_8, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf126 = extern_kernels.convolution(buf124, buf125, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf124
        buf127 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [branch_pool_9], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_37.run(buf105, buf127, 887808, grid=grid(887808), stream=stream0)
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg246_1
        buf129 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf106, arg202_1, arg203_1, arg204_1, arg205_1, buf113, arg217_1, arg218_1, arg219_1, arg220_1, buf126, arg242_1, arg243_1, arg244_1, arg245_1, buf128, arg247_1, arg248_1, arg249_1, arg250_1, buf129, 887808, grid=grid(887808), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        del buf106
        del buf113
        del buf126
        del buf128
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg251_1
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf129, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del arg256_1
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_105, branch7x7_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf132, arg257_1, arg258_1, arg259_1, arg260_1, 184960, grid=grid(184960), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        buf133 = reinterpret_tensor(buf122, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_105, branch7x7_6, x_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35.run(arg261_1, buf133, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg261_1
        # Topologically Sorted Source Nodes: [x_105, branch7x7_6, x_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf134 = extern_kernels.convolution(buf132, buf133, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del buf132
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_107, branch7x7_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf135, arg262_1, arg263_1, arg264_1, arg265_1, 184960, grid=grid(184960), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        buf136 = reinterpret_tensor(buf125, (192, 160, 7, 1), (1120, 1, 160, 160), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_107, branch7x7_7, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(arg266_1, buf136, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del arg266_1
        # Topologically Sorted Source Nodes: [x_107, branch7x7_7, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf137 = extern_kernels.convolution(buf135, buf136, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf135
        # Topologically Sorted Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf129, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del arg271_1
        buf139 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_111, branch7x7dbl_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf139, arg272_1, arg273_1, arg274_1, arg275_1, 184960, grid=grid(184960), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        buf140 = reinterpret_tensor(buf133, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_111, branch7x7dbl_10, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35.run(arg276_1, buf140, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg276_1
        # Topologically Sorted Source Nodes: [x_111, branch7x7dbl_10, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf141 = extern_kernels.convolution(buf139, buf140, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del buf139
        buf142 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_113, branch7x7dbl_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf142, arg277_1, arg278_1, arg279_1, arg280_1, 184960, grid=grid(184960), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        buf143 = reinterpret_tensor(buf140, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_113, branch7x7dbl_11, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35.run(arg281_1, buf143, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg281_1
        # Topologically Sorted Source Nodes: [x_113, branch7x7dbl_11, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf144 = extern_kernels.convolution(buf142, buf143, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del buf142
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_115, branch7x7dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf145, arg282_1, arg283_1, arg284_1, arg285_1, 184960, grid=grid(184960), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        buf146 = reinterpret_tensor(buf143, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_115, branch7x7dbl_12, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_35.run(arg286_1, buf146, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg286_1
        # Topologically Sorted Source Nodes: [x_115, branch7x7dbl_12, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf147 = extern_kernels.convolution(buf145, buf146, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 160, 17, 17), (46240, 1, 2720, 160))
        del buf145
        del buf146
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_117, branch7x7dbl_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf148, arg287_1, arg288_1, arg289_1, arg290_1, 184960, grid=grid(184960), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        buf149 = reinterpret_tensor(buf136, (192, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_117, branch7x7dbl_13, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(arg291_1, buf149, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del arg291_1
        # Topologically Sorted Source Nodes: [x_117, branch7x7dbl_13, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf150 = extern_kernels.convolution(buf148, buf149, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf148
        del buf149
        buf151 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [branch_pool_11], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_37.run(buf129, buf151, 887808, grid=grid(887808), stream=stream0)
        # Topologically Sorted Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg296_1
        buf153 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf130, arg252_1, arg253_1, arg254_1, arg255_1, buf137, arg267_1, arg268_1, arg269_1, arg270_1, buf150, arg292_1, arg293_1, arg294_1, arg295_1, buf152, arg297_1, arg298_1, arg299_1, arg300_1, buf153, 887808, grid=grid(887808), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        del buf130
        del buf137
        del buf150
        del buf152
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg301_1
        # Topologically Sorted Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf153, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg306_1
        buf156 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_125, branch7x7_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf156, arg307_1, arg308_1, arg309_1, arg310_1, 221952, grid=grid(221952), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        buf157 = empty_strided_cuda((192, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_125, branch7x7_9, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg311_1, buf157, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg311_1
        # Topologically Sorted Source Nodes: [x_125, branch7x7_9, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf158 = extern_kernels.convolution(buf156, buf157, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf156
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_127, branch7x7_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf159, arg312_1, arg313_1, arg314_1, arg315_1, 221952, grid=grid(221952), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        buf160 = reinterpret_tensor(buf157, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_127, branch7x7_10, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg316_1, buf160, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg316_1
        # Topologically Sorted Source Nodes: [x_127, branch7x7_10, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf161 = extern_kernels.convolution(buf159, buf160, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf159
        # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf153, arg321_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg321_1
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_131, branch7x7dbl_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf163, arg322_1, arg323_1, arg324_1, arg325_1, 221952, grid=grid(221952), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        buf164 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_131, branch7x7dbl_15, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg326_1, buf164, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg326_1
        # Topologically Sorted Source Nodes: [x_131, branch7x7dbl_15, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf165 = extern_kernels.convolution(buf163, buf164, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf163
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_133, branch7x7dbl_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf166, arg327_1, arg328_1, arg329_1, arg330_1, 221952, grid=grid(221952), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        buf167 = reinterpret_tensor(buf164, (192, 192, 1, 7), (1344, 1, 1344, 192), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_133, branch7x7dbl_16, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg331_1, buf167, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg331_1
        # Topologically Sorted Source Nodes: [x_133, branch7x7dbl_16, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf168 = extern_kernels.convolution(buf166, buf167, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf166
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_135, branch7x7dbl_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf169, arg332_1, arg333_1, arg334_1, arg335_1, 221952, grid=grid(221952), stream=stream0)
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        buf170 = reinterpret_tensor(buf167, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_135, branch7x7dbl_17, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg336_1, buf170, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg336_1
        # Topologically Sorted Source Nodes: [x_135, branch7x7dbl_17, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf171 = extern_kernels.convolution(buf169, buf170, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf169
        buf172 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_137, branch7x7dbl_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf172, arg337_1, arg338_1, arg339_1, arg340_1, 221952, grid=grid(221952), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        buf173 = reinterpret_tensor(buf170, (192, 192, 1, 7), (1344, 1, 1344, 192), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_137, branch7x7dbl_18, x_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg341_1, buf173, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg341_1
        # Topologically Sorted Source Nodes: [x_137, branch7x7dbl_18, x_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf174 = extern_kernels.convolution(buf172, buf173, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf172
        buf175 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [branch_pool_13], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_37.run(buf153, buf175, 887808, grid=grid(887808), stream=stream0)
        del buf153
        # Topologically Sorted Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg346_1
        buf177 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_33.run(buf154, arg302_1, arg303_1, arg304_1, arg305_1, buf161, arg317_1, arg318_1, arg319_1, arg320_1, buf174, arg342_1, arg343_1, arg344_1, arg345_1, buf176, arg347_1, arg348_1, arg349_1, arg350_1, buf177, 887808, grid=grid(887808), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        del buf154
        del buf161
        del buf174
        del buf176
        buf195 = empty_strided_cuda((4, 1280, 8, 8), (81920, 64, 8, 1), torch.float32)
        buf178 = reinterpret_tensor(buf195, (4, 768, 8, 8), (81920, 64, 8, 1), 32768)  # alias
        # Topologically Sorted Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_40.run(buf177, buf178, 256, 768, grid=grid(256, 768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf177, arg351_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg351_1
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_143, branch3x3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf180, arg352_1, arg353_1, arg354_1, arg355_1, 221952, grid=grid(221952), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        buf181 = empty_strided_cuda((320, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_143, branch3x3_1, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41.run(arg356_1, buf181, 61440, 9, grid=grid(61440, 9), stream=stream0)
        del arg356_1
        # Topologically Sorted Source Nodes: [x_143, branch3x3_1, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf182 = extern_kernels.convolution(buf180, buf181, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 320, 8, 8), (20480, 1, 2560, 320))
        del buf180
        del buf181
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf177, arg361_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del arg361_1
        del buf177
        buf184 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_147, branch7x7x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf184, arg362_1, arg363_1, arg364_1, arg365_1, 221952, grid=grid(221952), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        buf185 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_147, branch7x7x3, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg366_1, buf185, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg366_1
        # Topologically Sorted Source Nodes: [x_147, branch7x7x3, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf186 = extern_kernels.convolution(buf184, buf185, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf184
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_149, branch7x7x3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf187, arg367_1, arg368_1, arg369_1, arg370_1, 221952, grid=grid(221952), stream=stream0)
        del arg367_1
        del arg368_1
        del arg369_1
        del arg370_1
        buf188 = reinterpret_tensor(buf185, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [x_149, branch7x7x3_1, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg371_1, buf188, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg371_1
        # Topologically Sorted Source Nodes: [x_149, branch7x7x3_1, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf189 = extern_kernels.convolution(buf187, buf188, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 192, 17, 17), (55488, 1, 3264, 192))
        del buf187
        del buf188
        buf190 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_151, branch7x7x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf190, arg372_1, arg373_1, arg374_1, arg375_1, 221952, grid=grid(221952), stream=stream0)
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        buf191 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_151, branch7x7x3_2, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_42.run(arg376_1, buf191, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg376_1
        # Topologically Sorted Source Nodes: [x_151, branch7x7x3_2, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf192 = extern_kernels.convolution(buf190, buf191, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 192, 8, 8), (12288, 1, 1536, 192))
        del buf190
        del buf191
        buf193 = reinterpret_tensor(buf195, (4, 320, 8, 8), (81920, 64, 8, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_145, branch3x3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf182, arg357_1, arg358_1, arg359_1, arg360_1, buf193, 1280, 64, grid=grid(1280, 64), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        del buf182
        buf194 = reinterpret_tensor(buf195, (4, 192, 8, 8), (81920, 64, 8, 1), 20480)  # alias
        # Topologically Sorted Source Nodes: [x_153, branch7x7x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf192, arg377_1, arg378_1, arg379_1, arg380_1, buf194, 768, 64, grid=grid(768, 64), stream=stream0)
        del arg377_1
        del arg378_1
        del arg379_1
        del arg380_1
        del buf192
        buf196 = empty_strided_cuda((4, 1280, 8, 8), (81920, 1, 10240, 1280), torch.float32)
        buf198 = empty_strided_cuda((4, 1280, 8, 8), (81920, 1, 10240, 1280), torch.float32)
        buf206 = empty_strided_cuda((4, 1280, 8, 8), (81920, 1, 10240, 1280), torch.float32)
        buf217 = empty_strided_cuda((4, 1280, 8, 8), (81920, 1, 10240, 1280), torch.float32)
        # Topologically Sorted Source Nodes: [x_154, x_156, x_162, branch_pool_16], Original ATen: [aten.convolution, aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_convolution_45.run(buf195, buf196, buf198, buf206, buf217, 5120, 64, grid=grid(5120, 64), stream=stream0)
        del buf178
        del buf193
        del buf194
        del buf195
        # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg381_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 320, 8, 8), (20480, 1, 2560, 320))
        del arg381_1
        del buf196
        # Topologically Sorted Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg386_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 384, 8, 8), (24576, 1, 3072, 384))
        del arg386_1
        del buf198
        buf200 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_157, branch3x3_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf200, arg387_1, arg388_1, arg389_1, arg390_1, 98304, grid=grid(98304), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        buf201 = empty_strided_cuda((384, 384, 1, 3), (1152, 1, 1152, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(arg391_1, buf201, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg391_1
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf200, buf201, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf203 = reinterpret_tensor(buf201, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(arg396_1, buf203, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg396_1
        # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf200, buf203, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 384, 8, 8), (24576, 1, 3072, 384))
        del buf200
        buf221 = empty_strided_cuda((4, 2048, 8, 8), (131072, 64, 8, 1), torch.float32)
        buf205 = reinterpret_tensor(buf221, (4, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
        # Topologically Sorted Source Nodes: [branch3x3_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf202, arg392_1, arg393_1, arg394_1, arg395_1, buf204, arg397_1, arg398_1, arg399_1, arg400_1, buf205, 196608, grid=grid(196608), stream=stream0)
        del arg392_1
        del arg393_1
        del arg394_1
        del arg395_1
        del arg397_1
        del arg398_1
        del arg399_1
        del arg400_1
        del buf202
        del buf204
        # Topologically Sorted Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg401_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 448, 8, 8), (28672, 1, 3584, 448))
        del arg401_1
        del buf206
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_163, branch3x3dbl_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf208, arg402_1, arg403_1, arg404_1, arg405_1, 114688, grid=grid(114688), stream=stream0)
        del arg402_1
        del arg403_1
        del arg404_1
        del arg405_1
        buf209 = empty_strided_cuda((384, 448, 3, 3), (4032, 1, 1344, 448), torch.float32)
        # Topologically Sorted Source Nodes: [x_163, branch3x3dbl_12, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_50.run(arg406_1, buf209, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del arg406_1
        # Topologically Sorted Source Nodes: [x_163, branch3x3dbl_12, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 384, 8, 8), (24576, 1, 3072, 384))
        del buf208
        buf211 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [x_165, branch3x3dbl_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf211, arg407_1, arg408_1, arg409_1, arg410_1, 98304, grid=grid(98304), stream=stream0)
        del arg407_1
        del arg408_1
        del arg409_1
        del arg410_1
        buf212 = reinterpret_tensor(buf203, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_166], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(arg411_1, buf212, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg411_1
        # Topologically Sorted Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf214 = reinterpret_tensor(buf212, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(arg416_1, buf214, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg416_1
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf211, buf214, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 384, 8, 8), (24576, 1, 3072, 384))
        del buf211
        buf216 = reinterpret_tensor(buf221, (4, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
        # Topologically Sorted Source Nodes: [branch3x3dbl_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf213, arg412_1, arg413_1, arg414_1, arg415_1, buf215, arg417_1, arg418_1, arg419_1, arg420_1, buf216, 196608, grid=grid(196608), stream=stream0)
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        del arg417_1
        del arg418_1
        del arg419_1
        del arg420_1
        del buf213
        del buf215
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, arg421_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 192, 8, 8), (12288, 1, 1536, 192))
        del arg421_1
        del buf217
        buf219 = reinterpret_tensor(buf221, (4, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_155, branch1x1_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf197, arg382_1, arg383_1, arg384_1, arg385_1, buf219, 1280, 64, grid=grid(1280, 64), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        del buf197
        buf220 = reinterpret_tensor(buf221, (4, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
        # Topologically Sorted Source Nodes: [x_171, branch_pool_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf218, arg422_1, arg423_1, arg424_1, arg425_1, buf220, 768, 64, grid=grid(768, 64), stream=stream0)
        del arg422_1
        del arg423_1
        del arg424_1
        del arg425_1
        del buf218
        buf222 = empty_strided_cuda((4, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        buf224 = empty_strided_cuda((4, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        buf232 = empty_strided_cuda((4, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        buf243 = empty_strided_cuda((4, 2048, 8, 8), (131072, 1, 16384, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_172, x_174, x_180, branch_pool_18], Original ATen: [aten.convolution, aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_convolution_53.run(buf221, buf222, buf224, buf232, buf243, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del buf205
        del buf216
        del buf219
        del buf220
        del buf221
        # Topologically Sorted Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg426_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 320, 8, 8), (20480, 1, 2560, 320))
        del arg426_1
        del buf222
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, arg431_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 384, 8, 8), (24576, 1, 3072, 384))
        del arg431_1
        buf226 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_175, branch3x3_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf226, arg432_1, arg433_1, arg434_1, arg435_1, 98304, grid=grid(98304), stream=stream0)
        del arg432_1
        del arg433_1
        del arg434_1
        del arg435_1
        buf227 = reinterpret_tensor(buf214, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(arg436_1, buf227, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg436_1
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf229 = reinterpret_tensor(buf227, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(arg441_1, buf229, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg441_1
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf226, buf229, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 384, 8, 8), (24576, 1, 3072, 384))
        del buf226
        buf247 = reinterpret_tensor(buf224, (4, 2048, 8, 8), (131072, 64, 8, 1), 0); del buf224  # reuse
        buf231 = reinterpret_tensor(buf247, (4, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
        # Topologically Sorted Source Nodes: [branch3x3_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf228, arg437_1, arg438_1, arg439_1, arg440_1, buf230, arg442_1, arg443_1, arg444_1, arg445_1, buf231, 196608, grid=grid(196608), stream=stream0)
        del arg437_1
        del arg438_1
        del arg439_1
        del arg440_1
        del arg442_1
        del arg443_1
        del arg444_1
        del arg445_1
        del buf228
        del buf230
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, arg446_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 448, 8, 8), (28672, 1, 3584, 448))
        del arg446_1
        del buf232
        buf234 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_181, branch3x3dbl_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf234, arg447_1, arg448_1, arg449_1, arg450_1, 114688, grid=grid(114688), stream=stream0)
        del arg447_1
        del arg448_1
        del arg449_1
        del arg450_1
        buf235 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_181, branch3x3dbl_15, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_50.run(arg451_1, buf235, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del arg451_1
        # Topologically Sorted Source Nodes: [x_181, branch3x3dbl_15, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf236 = extern_kernels.convolution(buf234, buf235, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 384, 8, 8), (24576, 1, 3072, 384))
        del buf234
        del buf235
        buf237 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_183, branch3x3dbl_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf237, arg452_1, arg453_1, arg454_1, arg455_1, 98304, grid=grid(98304), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        buf238 = reinterpret_tensor(buf229, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(arg456_1, buf238, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg456_1
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf237, buf238, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf240 = reinterpret_tensor(buf238, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_47.run(arg461_1, buf240, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg461_1
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf237, buf240, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 384, 8, 8), (24576, 1, 3072, 384))
        del buf237
        del buf240
        buf242 = reinterpret_tensor(buf247, (4, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
        # Topologically Sorted Source Nodes: [branch3x3dbl_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_48.run(buf239, arg457_1, arg458_1, arg459_1, arg460_1, buf241, arg462_1, arg463_1, arg464_1, arg465_1, buf242, 196608, grid=grid(196608), stream=stream0)
        del arg457_1
        del arg458_1
        del arg459_1
        del arg460_1
        del arg462_1
        del arg463_1
        del arg464_1
        del arg465_1
        del buf239
        del buf241
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, arg466_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 192, 8, 8), (12288, 1, 1536, 192))
        del arg466_1
        del buf243
        buf245 = reinterpret_tensor(buf247, (4, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_173, branch1x1_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_51.run(buf223, arg427_1, arg428_1, arg429_1, arg430_1, buf245, 1280, 64, grid=grid(1280, 64), stream=stream0)
        del arg427_1
        del arg428_1
        del arg429_1
        del arg430_1
        del buf223
        buf246 = reinterpret_tensor(buf247, (4, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
        # Topologically Sorted Source Nodes: [x_189, branch_pool_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf244, arg467_1, arg468_1, arg469_1, arg470_1, buf246, 768, 64, grid=grid(768, 64), stream=stream0)
        del arg467_1
        del arg468_1
        del arg469_1
        del arg470_1
        del buf244
        buf248 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 8192, 8192), torch.float32)
        buf249 = reinterpret_tensor(buf248, (4, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_54.run(buf249, buf247, 8192, 64, grid=grid(8192), stream=stream0)
        del buf231
        del buf242
        del buf245
        del buf246
        del buf247
    return (buf249, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
