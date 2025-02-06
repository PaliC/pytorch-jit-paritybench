# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/h6/ch6u7ggmfki6iusqwe7hvxbnooqjsnrwytiqwqhnn55djsqsqyvy.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten._unsafe_index, aten.sub, aten.add]
# Source node to ATen node mapping:
#   x => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_2, add_3, add_4, clamp_max_2, clamp_max_3, clamp_min, clamp_min_2, clamp_min_3, convert_element_type, convert_element_type_1, convert_element_type_3, iota, mul, mul_2, mul_3, mul_4, sub, sub_1, sub_2, sub_3, sub_4
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (126,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 0.024), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul, 0.0), kwargs = {})
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min, torch.int64), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_1, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
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
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_4), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 254016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 126) % 126)
    x0 = (xindex % 126)
    x2 = xindex // 15876
    x3 = (xindex % 15876)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.024
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = x0
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = triton_helpers.maximum(tmp13, tmp4)
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + 4*tmp10 + 16*x2), xmask, eviction_policy='evict_last')
    tmp17 = tmp15 + tmp7
    tmp18 = triton_helpers.minimum(tmp17, tmp9)
    tmp19 = tl.load(in_ptr0 + (tmp18 + 4*tmp10 + 16*x2), xmask, eviction_policy='evict_last')
    tmp20 = tmp19 - tmp16
    tmp21 = tmp15.to(tl.float32)
    tmp22 = tmp14 - tmp21
    tmp23 = triton_helpers.maximum(tmp22, tmp4)
    tmp24 = 1.0
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp20 * tmp25
    tmp27 = tmp16 + tmp26
    tmp28 = tl.load(in_ptr0 + (tmp15 + 4*tmp6 + 16*x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (tmp18 + 4*tmp6 + 16*x2), xmask, eviction_policy='evict_last')
    tmp30 = tmp29 - tmp28
    tmp31 = tmp30 * tmp25
    tmp32 = tmp28 + tmp31
    tmp33 = tmp27 - tmp32
    tmp34 = tmp6.to(tl.float32)
    tmp35 = tmp5 - tmp34
    tmp36 = triton_helpers.maximum(tmp35, tmp4)
    tmp37 = triton_helpers.minimum(tmp36, tmp24)
    tmp38 = tmp33 * tmp37
    tmp39 = tmp32 + tmp38
    tl.store(out_ptr1 + (x4), tmp39, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eg/cegdgwm5dbvaar6eu2z7om563s6zs7ypd26a7yhewv45vlmwnn6k.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   input_2 => gt, mul_5, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul_5), kwargs = {})
triton_poi_fused_leaky_relu_1 = async_compile.triton('triton_poi_fused_leaky_relu_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 492032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ob/cobuviowwpt7wddig6g2jnf4zw7wbfsezc3vamu2evia3guej4au.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   input_4 => gt_1, mul_6, where_1
# Graph fragment:
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.2), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_1, %mul_6), kwargs = {})
triton_poi_fused_leaky_relu_2 = async_compile.triton('triton_poi_fused_leaky_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pc/cpc7khfx2w42v4vgcqodjrx7y45hduhlo5s5c26rwgojelfsuva5.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   input_6 => gt_2, mul_7, where_2
# Graph fragment:
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 0.2), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_2, %mul_7), kwargs = {})
triton_poi_fused_leaky_relu_3 = async_compile.triton('triton_poi_fused_leaky_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_3(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 373248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5z/c5z4tahumtwg3khnxmv4wimfvajts6oteesaijllraag3pfcr2os.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.sigmoid]
# Source node to ATen node mapping:
#   input_7 => convolution_3
#   input_8 => sigmoid
# Graph fragment:
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_2, %primals_5, %primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_3,), kwargs = {})
triton_poi_fused_convolution_sigmoid_4 = async_compile.triton('triton_poi_fused_convolution_sigmoid_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_sigmoid_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (32, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (64, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(primals_4, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_5, (1, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_6, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 4, 126, 126), (63504, 15876, 126, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten._unsafe_index, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0.run(primals_1, buf1, 254016, grid=grid(254016), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_2, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 62, 62), (123008, 3844, 62, 1))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_1.run(buf3, 492032, grid=grid(492032), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_3, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 64, 30, 30), (57600, 900, 30, 1))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_2.run(buf5, 230400, grid=grid(230400), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 128, 27, 27), (93312, 729, 27, 1))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_3.run(buf7, 373248, grid=grid(373248), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 1, 24, 24), (576, 576, 24, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_sigmoid_4.run(buf9, primals_6, 2304, grid=grid(2304), stream=stream0)
        del primals_6
    return (buf9, primals_2, primals_3, primals_4, primals_5, buf1, buf3, buf5, buf7, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 32, 4, 4), (512, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
