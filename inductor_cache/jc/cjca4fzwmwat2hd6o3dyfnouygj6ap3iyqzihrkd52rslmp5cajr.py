# AOT ID: ['63_forward']
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


# kernel path: inductor_cache/bl/cblvw332w67t4k2eyb2mg3im7ahlxjo6plh6gy5lpzhsbkhzez4o.py
# Topologically Sorted Source Nodes: [conv2d, x], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv2d => convolution
#   x => add_1, mul_1, mul_2, sub
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [4, 4], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 24)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/wp/cwp6atwx4b3e2wdkwquqy753gucfsxa7lxxmqxr57kutq2y7cjvk.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %neg], 1), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%cat, 0.0), kwargs = {})
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%cat, 6.0), kwargs = {})
#   %bitwise_or_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_1, %ge_1), kwargs = {})
triton_poi_fused_cat_hardtanh_backward_1 = async_compile.triton('triton_poi_fused_cat_hardtanh_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_hardtanh_backward_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_hardtanh_backward_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 48)
    x0 = (xindex % 256)
    x2 = xindex // 12288
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 6144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 48, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 256*((-24) + x1) + 6144*x2), tmp6, other=0.0)
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp6, tmp10, tmp11)
    tmp13 = tl.where(tmp4, tmp5, tmp12)
    tmp14 = 0.0
    tmp15 = tmp13 <= tmp14
    tmp16 = 6.0
    tmp17 = tmp13 >= tmp16
    tmp18 = tmp15 | tmp17
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/ud/cud56gvh5oygzz2b4uix2jfq4ysgeyuuxricwl25upc4kda6bxkp.py
# Topologically Sorted Source Nodes: [out, input_1], Original ATen: [aten.cat, aten.hardtanh]
# Source node to ATen node mapping:
#   input_1 => clamp_max, clamp_min
#   out => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %neg], 1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%cat, 0.0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused_cat_hardtanh_2 = async_compile.triton('triton_poi_fused_cat_hardtanh_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_hardtanh_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_hardtanh_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 48)
    x0 = (xindex % 256)
    x2 = xindex // 12288
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 6144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 48, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 256*((-24) + x1) + 6144*x2), tmp6, other=0.0)
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp6, tmp10, tmp11)
    tmp13 = tl.where(tmp4, tmp5, tmp12)
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/g6/cg6eeosgk7dowyuvw36zddrblqdtz3yzbb2vsssyxekmy52y4eef.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_2 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x3 = xindex // 8
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-17) + 2*x0 + 32*x3), tmp10, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-16) + 2*x0 + 32*x3), tmp16, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-15) + 2*x0 + 32*x3), tmp23, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 32*x3), tmp30, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 32*x3), tmp33, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x3), tmp36, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (15 + 2*x0 + 32*x3), tmp43, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x3), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x3), tmp49, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tl.store(out_ptr0 + (x4), tmp51, None)
    tl.store(out_ptr1 + (x4), tmp76, None)
''', device_str='cuda')


# kernel path: inductor_cache/dp/cdpuppz5zdjanid7xykhnnfb7gbxcblvqtiu7g67zggo2nvuoaxm.py
# Topologically Sorted Source Nodes: [conv2d_1, x_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   x_1 => add_3, mul_4, mul_5, sub_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %primals_8, %primals_9, [2, 2], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/jd/cjdgdemyjd4lhlcdoei2ujiwz44rwu4fa55mwclswcmln4afgwry.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat, aten.hardtanh_backward]
# Source node to ATen node mapping:
#   out_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_3, %neg_1], 1), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%cat_1, 0.0), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%cat_1, 6.0), kwargs = {})
#   %bitwise_or : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le, %ge), kwargs = {})
triton_poi_fused_cat_hardtanh_backward_5 = async_compile.triton('triton_poi_fused_cat_hardtanh_backward_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_hardtanh_backward_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_hardtanh_backward_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 128)
    x0 = (xindex % 16)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 1024*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 16*((-64) + x1) + 1024*x2), tmp6, other=0.0)
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp6, tmp10, tmp11)
    tmp13 = tl.where(tmp4, tmp5, tmp12)
    tmp14 = 0.0
    tmp15 = tmp13 <= tmp14
    tmp16 = 6.0
    tmp17 = tmp13 >= tmp16
    tmp18 = tmp15 | tmp17
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/un/cunel76argx2ponhprrjtimys75d5qvlbdmygxykxajmtghrpzs7.py
# Topologically Sorted Source Nodes: [out_1, input_3], Original ATen: [aten.cat, aten.hardtanh]
# Source node to ATen node mapping:
#   input_3 => clamp_max_1, clamp_min_1
#   out_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_3, %neg_1], 1), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%cat_1, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6.0), kwargs = {})
triton_poi_fused_cat_hardtanh_6 = async_compile.triton('triton_poi_fused_cat_hardtanh_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_hardtanh_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_hardtanh_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 128)
    x0 = (xindex % 16)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 1024*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 16*((-64) + x1) + 1024*x2), tmp6, other=0.0)
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp6, tmp10, tmp11)
    tmp13 = tl.where(tmp4, tmp5, tmp12)
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/wj/cwjoa6ygmnkvo52rtqhofrzqgtmdpltcc4z2qq7qbjnzcguudvaz.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_4 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=5] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_7 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x3 = xindex // 2
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + 2*x0 + 8*x3), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + 2*x0 + 8*x3), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + 2*x0 + 8*x3), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 8*x3), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 8*x3), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x3), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + 2*x0 + 8*x3), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x3), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x3), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tl.store(out_ptr0 + (x4), tmp51, xmask)
    tl.store(out_ptr1 + (x4), tmp76, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sk/csk7icg2wi2r6zfxrjvtvmteor2q4axz2a6bo6pj33czlgdr62rs.py
# Topologically Sorted Source Nodes: [conv2d_4, batch_norm_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_4 => add_9, mul_13, mul_14, sub_4
#   conv2d_4 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_26, %primals_27, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 24)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n4/cn4iccj2g2kabdswapawq2hysgyjiegc7vi4wckvml2gjnkowkpd.py
# Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_2 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7k/c7kgatexnfvpzb5c2zjp2srrkjcvzhn3fwa5rbutelbx7j7yw5xj.py
# Topologically Sorted Source Nodes: [conv2d_7, batch_norm_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_7 => add_15, mul_22, mul_23, sub_7
#   conv2d_7 => convolution_7
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_13, %primals_44, %primals_45, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jq/cjqo3x5yafaz2yro5rthlisd3pzcwhc7jlmw6h3g7u5frq5s4vds.py
# Topologically Sorted Source Nodes: [max_pool2d_2], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   max_pool2d_2 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_11 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_11(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-3) + x4), tmp10 & xmask, other=float("-inf"))
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-2) + x4), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-1) + x4), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (1 + x4), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (2 + x4), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (3 + x4), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tl.store(out_ptr0 + (x4), tmp51, xmask)
    tl.store(out_ptr1 + (x4), tmp76, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h6/ch6fkyotq2dennhokcvo4ziepctzlu4ynslz74g47v7nwfljz7ht.py
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%add_5, %add_7, %add_11, %add_17], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 128)
    x0 = (xindex % 4)
    x2 = xindex // 512
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 128*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 64, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr5 + (x0 + 4*((-32) + x1) + 128*x2), tmp26 & xmask, other=0.0)
    tmp28 = tl.load(in_ptr6 + ((-32) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.load(in_ptr7 + ((-32) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp39 = tl.load(in_ptr8 + ((-32) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.load(in_ptr9 + ((-32) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp26, tmp42, tmp43)
    tmp45 = tmp0 >= tmp24
    tmp46 = tl.full([1], 96, tl.int64)
    tmp47 = tmp0 < tmp46
    tmp48 = tmp45 & tmp47
    tmp49 = tl.load(in_ptr10 + (x0 + 4*((-64) + x1) + 128*x2), tmp48 & xmask, other=0.0)
    tmp50 = tl.load(in_ptr11 + ((-64) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 - tmp50
    tmp52 = tl.load(in_ptr12 + ((-64) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = 1e-05
    tmp54 = tmp52 + tmp53
    tmp55 = libdevice.sqrt(tmp54)
    tmp56 = tl.full([1], 1, tl.int32)
    tmp57 = tmp56 / tmp55
    tmp58 = 1.0
    tmp59 = tmp57 * tmp58
    tmp60 = tmp51 * tmp59
    tmp61 = tl.load(in_ptr13 + ((-64) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp60 * tmp61
    tmp63 = tl.load(in_ptr14 + ((-64) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp62 + tmp63
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp48, tmp64, tmp65)
    tmp67 = tmp0 >= tmp46
    tmp68 = tl.full([1], 128, tl.int64)
    tmp69 = tmp0 < tmp68
    tmp70 = tl.load(in_ptr15 + (x0 + 4*((-96) + x1) + 128*x2), tmp67 & xmask, other=0.0)
    tmp71 = tl.load(in_ptr16 + ((-96) + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 - tmp71
    tmp73 = tl.load(in_ptr17 + ((-96) + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = 1e-05
    tmp75 = tmp73 + tmp74
    tmp76 = libdevice.sqrt(tmp75)
    tmp77 = tl.full([1], 1, tl.int32)
    tmp78 = tmp77 / tmp76
    tmp79 = 1.0
    tmp80 = tmp78 * tmp79
    tmp81 = tmp72 * tmp80
    tmp82 = tl.load(in_ptr18 + ((-96) + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp81 * tmp82
    tmp84 = tl.load(in_ptr19 + ((-96) + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 + tmp84
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp67, tmp85, tmp86)
    tmp88 = tl.where(tmp48, tmp66, tmp87)
    tmp89 = tl.where(tmp26, tmp44, tmp88)
    tmp90 = tl.where(tmp4, tmp22, tmp89)
    tl.store(out_ptr0 + (x3), tmp90, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2s/c2sscnlxstmxwp54seqsowbwg5qzf7qdmcwc5bgc76erraphhkfm.py
# Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_23 => convolution_23
# Graph fragment:
#   %convolution_23 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_4, %primals_140, %primals_141, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_13 = async_compile.triton('triton_poi_fused_convolution_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ag/cagv4535vioewi34vupy3kty3t3tkwwtwsw4txykvqyil5y7gk6l.py
# Topologically Sorted Source Nodes: [out2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out2 => convolution_24
# Graph fragment:
#   %convolution_24 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_23, %primals_142, %primals_143, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yh/cyhbocnnrr4lwt4iffzk6imccmbjzmvwmlksuf5l3wnhh72p7zy6.py
# Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_25 => convolution_25
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_24, %primals_144, %primals_145, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/if/cifz5uruqwpinmeixsoqht3gkhs2wua4ovk2c4fovsqq6inmkfhj.py
# Topologically Sorted Source Nodes: [locs], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   locs => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %view_1, %view_2], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 344)
    x1 = xindex // 344
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 336, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*(((x0) % 84)) + 336*x1 + ((((x0) // 84) % 4))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (((x0) % 84)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 340, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (4*x1 + ((-336) + x0)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-336) + x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 344, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + (4*x1 + ((-340) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + ((-340) + x0), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (24, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (24, ), (1, ))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (24, ), (1, ))
    assert_size_stride(primals_7, (24, ), (1, ))
    assert_size_stride(primals_8, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (32, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (24, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_27, (24, ), (1, ))
    assert_size_stride(primals_28, (24, ), (1, ))
    assert_size_stride(primals_29, (24, ), (1, ))
    assert_size_stride(primals_30, (24, ), (1, ))
    assert_size_stride(primals_31, (24, ), (1, ))
    assert_size_stride(primals_32, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_33, (32, ), (1, ))
    assert_size_stride(primals_34, (32, ), (1, ))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_37, (32, ), (1, ))
    assert_size_stride(primals_38, (24, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_39, (24, ), (1, ))
    assert_size_stride(primals_40, (24, ), (1, ))
    assert_size_stride(primals_41, (24, ), (1, ))
    assert_size_stride(primals_42, (24, ), (1, ))
    assert_size_stride(primals_43, (24, ), (1, ))
    assert_size_stride(primals_44, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, ), (1, ))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_48, (32, ), (1, ))
    assert_size_stride(primals_49, (32, ), (1, ))
    assert_size_stride(primals_50, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_52, (32, ), (1, ))
    assert_size_stride(primals_53, (32, ), (1, ))
    assert_size_stride(primals_54, (32, ), (1, ))
    assert_size_stride(primals_55, (32, ), (1, ))
    assert_size_stride(primals_56, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_57, (32, ), (1, ))
    assert_size_stride(primals_58, (32, ), (1, ))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (32, ), (1, ))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_63, (32, ), (1, ))
    assert_size_stride(primals_64, (32, ), (1, ))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (32, ), (1, ))
    assert_size_stride(primals_67, (32, ), (1, ))
    assert_size_stride(primals_68, (24, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_69, (24, ), (1, ))
    assert_size_stride(primals_70, (24, ), (1, ))
    assert_size_stride(primals_71, (24, ), (1, ))
    assert_size_stride(primals_72, (24, ), (1, ))
    assert_size_stride(primals_73, (24, ), (1, ))
    assert_size_stride(primals_74, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_75, (32, ), (1, ))
    assert_size_stride(primals_76, (32, ), (1, ))
    assert_size_stride(primals_77, (32, ), (1, ))
    assert_size_stride(primals_78, (32, ), (1, ))
    assert_size_stride(primals_79, (32, ), (1, ))
    assert_size_stride(primals_80, (24, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_81, (24, ), (1, ))
    assert_size_stride(primals_82, (24, ), (1, ))
    assert_size_stride(primals_83, (24, ), (1, ))
    assert_size_stride(primals_84, (24, ), (1, ))
    assert_size_stride(primals_85, (24, ), (1, ))
    assert_size_stride(primals_86, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_87, (32, ), (1, ))
    assert_size_stride(primals_88, (32, ), (1, ))
    assert_size_stride(primals_89, (32, ), (1, ))
    assert_size_stride(primals_90, (32, ), (1, ))
    assert_size_stride(primals_91, (32, ), (1, ))
    assert_size_stride(primals_92, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_93, (32, ), (1, ))
    assert_size_stride(primals_94, (32, ), (1, ))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, ), (1, ))
    assert_size_stride(primals_97, (32, ), (1, ))
    assert_size_stride(primals_98, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_101, (32, ), (1, ))
    assert_size_stride(primals_102, (32, ), (1, ))
    assert_size_stride(primals_103, (32, ), (1, ))
    assert_size_stride(primals_104, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, ), (1, ))
    assert_size_stride(primals_107, (32, ), (1, ))
    assert_size_stride(primals_108, (32, ), (1, ))
    assert_size_stride(primals_109, (32, ), (1, ))
    assert_size_stride(primals_110, (24, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_111, (24, ), (1, ))
    assert_size_stride(primals_112, (24, ), (1, ))
    assert_size_stride(primals_113, (24, ), (1, ))
    assert_size_stride(primals_114, (24, ), (1, ))
    assert_size_stride(primals_115, (24, ), (1, ))
    assert_size_stride(primals_116, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_117, (32, ), (1, ))
    assert_size_stride(primals_118, (32, ), (1, ))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (32, ), (1, ))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (24, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_123, (24, ), (1, ))
    assert_size_stride(primals_124, (24, ), (1, ))
    assert_size_stride(primals_125, (24, ), (1, ))
    assert_size_stride(primals_126, (24, ), (1, ))
    assert_size_stride(primals_127, (24, ), (1, ))
    assert_size_stride(primals_128, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_129, (32, ), (1, ))
    assert_size_stride(primals_130, (32, ), (1, ))
    assert_size_stride(primals_131, (32, ), (1, ))
    assert_size_stride(primals_132, (32, ), (1, ))
    assert_size_stride(primals_133, (32, ), (1, ))
    assert_size_stride(primals_134, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_135, (32, ), (1, ))
    assert_size_stride(primals_136, (32, ), (1, ))
    assert_size_stride(primals_137, (32, ), (1, ))
    assert_size_stride(primals_138, (32, ), (1, ))
    assert_size_stride(primals_139, (32, ), (1, ))
    assert_size_stride(primals_140, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_144, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_147, (256, ), (1, ))
    assert_size_stride(primals_148, (84, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_149, (84, ), (1, ))
    assert_size_stride(primals_150, (84, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_151, (84, ), (1, ))
    assert_size_stride(primals_152, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_153, (4, ), (1, ))
    assert_size_stride(primals_154, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_155, (4, ), (1, ))
    assert_size_stride(primals_156, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_157, (4, ), (1, ))
    assert_size_stride(primals_158, (4, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_159, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(4, 4), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 24, 16, 16), (6144, 256, 16, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 24, 16, 16), (6144, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d, x], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_0.run(buf1, primals_2, primals_4, primals_5, primals_6, primals_7, buf2, 24576, grid=grid(24576), stream=stream0)
        del primals_2
        del primals_7
        buf89 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_hardtanh_backward_1.run(buf2, buf89, 49152, grid=grid(49152), stream=stream0)
        buf3 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out, input_1], Original ATen: [aten.cat, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_hardtanh_2.run(buf2, buf3, 49152, grid=grid(49152), stream=stream0)
        del buf2
        buf4 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_3.run(buf3, buf4, buf5, 12288, grid=grid(12288), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf4, primals_8, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_1, x_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_4.run(buf7, primals_9, primals_10, primals_11, primals_12, primals_13, buf8, 4096, grid=grid(4096), stream=stream0)
        del primals_13
        del primals_9
        buf88 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.cat, aten.hardtanh_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_hardtanh_backward_5.run(buf8, buf88, 8192, grid=grid(8192), stream=stream0)
        buf9 = empty_strided_cuda((4, 128, 4, 4), (2048, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, input_3], Original ATen: [aten.cat, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_hardtanh_6.run(buf8, buf9, 8192, grid=grid(8192), stream=stream0)
        del buf8
        buf10 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        buf11 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_7.run(buf9, buf10, buf11, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf10, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 24, 2, 2), (96, 4, 2, 1))
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided_cuda((4, 24, 2, 2), (96, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_4, batch_norm_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_8.run(buf19, primals_27, primals_28, primals_29, primals_30, primals_31, buf20, 384, grid=grid(384), stream=stream0)
        del primals_27
        del primals_31
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf10, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 24, 2, 2), (96, 4, 2, 1))
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 24, 2, 2), (96, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_6, batch_norm_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_8.run(buf24, primals_39, primals_40, primals_41, primals_42, primals_43, buf25, 384, grid=grid(384), stream=stream0)
        del primals_39
        del primals_43
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf10, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 32, 2, 2), (128, 4, 2, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf13, primals_15, 512, grid=grid(512), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 32, 2, 2), (128, 4, 2, 1))
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf22, primals_33, 512, grid=grid(512), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 32, 2, 2), (128, 4, 2, 1))
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_7, batch_norm_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_10.run(buf27, primals_45, primals_46, primals_47, primals_48, primals_49, buf28, 512, grid=grid(512), stream=stream0)
        del primals_45
        del primals_49
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 32, 2, 2), (128, 4, 2, 1))
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf30, primals_51, 512, grid=grid(512), stream=stream0)
        del primals_51
        buf14 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [max_pool2d_2], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_11.run(buf10, buf14, buf15, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf14, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 32, 2, 2), (128, 4, 2, 1))
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf17, primals_21, 512, grid=grid(512), stream=stream0)
        del primals_21
        buf31 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf13, primals_16, primals_17, primals_18, primals_19, buf17, primals_22, primals_23, primals_24, primals_25, buf22, primals_34, primals_35, primals_36, primals_37, buf30, primals_52, primals_53, primals_54, primals_55, buf31, 2048, grid=grid(2048), stream=stream0)
        del primals_19
        del primals_25
        del primals_37
        del primals_55
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf31, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 24, 2, 2), (96, 4, 2, 1))
        buf39 = buf38; del buf38  # reuse
        buf40 = empty_strided_cuda((4, 24, 2, 2), (96, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_11, batch_norm_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_8.run(buf39, primals_69, primals_70, primals_71, primals_72, primals_73, buf40, 384, grid=grid(384), stream=stream0)
        del primals_69
        del primals_73
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf31, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 24, 2, 2), (96, 4, 2, 1))
        buf44 = buf43; del buf43  # reuse
        buf45 = empty_strided_cuda((4, 24, 2, 2), (96, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_13, batch_norm_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_8.run(buf44, primals_81, primals_82, primals_83, primals_84, primals_85, buf45, 384, grid=grid(384), stream=stream0)
        del primals_81
        del primals_85
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 32, 2, 2), (128, 4, 2, 1))
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf33, primals_57, 512, grid=grid(512), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 32, 2, 2), (128, 4, 2, 1))
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf42, primals_75, 512, grid=grid(512), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 32, 2, 2), (128, 4, 2, 1))
        buf47 = buf46; del buf46  # reuse
        buf48 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_14, batch_norm_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_10.run(buf47, primals_87, primals_88, primals_89, primals_90, primals_91, buf48, 512, grid=grid(512), stream=stream0)
        del primals_87
        del primals_91
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 32, 2, 2), (128, 4, 2, 1))
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf50, primals_93, 512, grid=grid(512), stream=stream0)
        del primals_93
        buf34 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        buf35 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [max_pool2d_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_11.run(buf31, buf34, buf35, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf34, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 32, 2, 2), (128, 4, 2, 1))
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf37, primals_63, 512, grid=grid(512), stream=stream0)
        del primals_63
        buf51 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf33, primals_58, primals_59, primals_60, primals_61, buf37, primals_64, primals_65, primals_66, primals_67, buf42, primals_76, primals_77, primals_78, primals_79, buf50, primals_94, primals_95, primals_96, primals_97, buf51, 2048, grid=grid(2048), stream=stream0)
        del primals_61
        del primals_67
        del primals_79
        del primals_97
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf51, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 24, 2, 2), (96, 4, 2, 1))
        buf59 = buf58; del buf58  # reuse
        buf60 = empty_strided_cuda((4, 24, 2, 2), (96, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_18, batch_norm_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_8.run(buf59, primals_111, primals_112, primals_113, primals_114, primals_115, buf60, 384, grid=grid(384), stream=stream0)
        del primals_111
        del primals_115
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf51, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 24, 2, 2), (96, 4, 2, 1))
        buf64 = buf63; del buf63  # reuse
        buf65 = empty_strided_cuda((4, 24, 2, 2), (96, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_20, batch_norm_20], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_8.run(buf64, primals_123, primals_124, primals_125, primals_126, primals_127, buf65, 384, grid=grid(384), stream=stream0)
        del primals_123
        del primals_127
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 32, 2, 2), (128, 4, 2, 1))
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf53, primals_99, 512, grid=grid(512), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 32, 2, 2), (128, 4, 2, 1))
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf62, primals_117, 512, grid=grid(512), stream=stream0)
        del primals_117
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 32, 2, 2), (128, 4, 2, 1))
        buf67 = buf66; del buf66  # reuse
        buf68 = empty_strided_cuda((4, 32, 2, 2), (128, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_21, batch_norm_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_10.run(buf67, primals_129, primals_130, primals_131, primals_132, primals_133, buf68, 512, grid=grid(512), stream=stream0)
        del primals_129
        del primals_133
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_134, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 32, 2, 2), (128, 4, 2, 1))
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf70, primals_135, 512, grid=grid(512), stream=stream0)
        del primals_135
        buf54 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        buf55 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.int8)
        # Topologically Sorted Source Nodes: [max_pool2d_4], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_11.run(buf51, buf54, buf55, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf54, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 32, 2, 2), (128, 4, 2, 1))
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf57, primals_105, 512, grid=grid(512), stream=stream0)
        del primals_105
        buf71 = empty_strided_cuda((4, 128, 2, 2), (512, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(buf53, primals_100, primals_101, primals_102, primals_103, buf57, primals_106, primals_107, primals_108, primals_109, buf62, primals_118, primals_119, primals_120, primals_121, buf70, primals_136, primals_137, primals_138, primals_139, buf71, 2048, grid=grid(2048), stream=stream0)
        del primals_103
        del primals_109
        del primals_121
        del primals_139
        # Topologically Sorted Source Nodes: [loc1], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf71, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 84, 2, 2), (336, 4, 2, 1))
        # Topologically Sorted Source Nodes: [conf1], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf71, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 84, 2, 2), (336, 4, 2, 1))
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 128, 2, 2), (512, 4, 2, 1))
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_13.run(buf73, primals_141, 2048, grid=grid(2048), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [out2], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_142, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 256, 1, 1), (256, 1, 1, 1))
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [out2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_14.run(buf75, primals_143, 1024, grid=grid(1024), stream=stream0)
        del primals_143
        # Topologically Sorted Source Nodes: [loc2], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf75, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 4, 1, 1), (4, 1, 1, 1))
        # Topologically Sorted Source Nodes: [conf2], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf75, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 4, 1, 1), (4, 1, 1, 1))
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 128, 1, 1), (128, 1, 1, 1))
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_15.run(buf77, primals_145, 512, grid=grid(512), stream=stream0)
        del primals_145
        # Topologically Sorted Source Nodes: [out3], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_146, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 256, 1, 1), (256, 1, 1, 1))
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [out3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_14.run(buf79, primals_147, 1024, grid=grid(1024), stream=stream0)
        del primals_147
        # Topologically Sorted Source Nodes: [loc3], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf79, primals_156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 4, 1, 1), (4, 1, 1, 1))
        # Topologically Sorted Source Nodes: [conf3], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf79, primals_158, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 4, 1, 1), (4, 1, 1, 1))
        buf86 = empty_strided_cuda((4, 344), (344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [locs], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf80, primals_149, buf82, primals_153, buf84, primals_157, buf86, 1376, grid=grid(1376), stream=stream0)
        del buf80
        del buf82
        del buf84
        del primals_149
        del primals_153
        del primals_157
        buf87 = empty_strided_cuda((4, 344), (344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [confs], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf81, primals_151, buf83, primals_155, buf85, primals_159, buf87, 1376, grid=grid(1376), stream=stream0)
        del buf81
        del buf83
        del buf85
        del primals_151
        del primals_155
        del primals_159
    return (reinterpret_tensor(buf86, (4, 86, 4), (344, 4, 1), 0), reinterpret_tensor(buf87, (344, 4), (4, 1), 0), primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_28, primals_29, primals_30, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_58, primals_59, primals_60, primals_62, primals_64, primals_65, primals_66, primals_68, primals_70, primals_71, primals_72, primals_74, primals_76, primals_77, primals_78, primals_80, primals_82, primals_83, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_98, primals_100, primals_101, primals_102, primals_104, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_116, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_128, primals_130, primals_131, primals_132, primals_134, primals_136, primals_137, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, buf1, buf3, buf4, buf5, buf7, buf9, buf10, buf11, buf13, buf14, buf15, buf17, buf19, buf20, buf22, buf24, buf25, buf27, buf28, buf30, buf31, buf33, buf34, buf35, buf37, buf39, buf40, buf42, buf44, buf45, buf47, buf48, buf50, buf51, buf53, buf54, buf55, buf57, buf59, buf60, buf62, buf64, buf65, buf67, buf68, buf70, buf71, buf73, buf75, buf77, buf79, buf88, buf89, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((24, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((24, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((24, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((24, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((24, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((84, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((84, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((4, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
