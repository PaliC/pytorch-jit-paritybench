# AOT ID: ['157_forward']
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


# kernel path: inductor_cache/ya/cyaooxpe3q2p5zpacbknaybzfys7d5nj3ejnazgydgivovbi73ap.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
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


# kernel path: inductor_cache/gb/cgbelhqkjgxe3ifzkqogzdmi7zklenhj4f24rvxb75lvfpod6wyk.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   out => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_1 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x3 = xindex // 32
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-65) + 2*x0 + 128*x3), tmp10, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-64) + 2*x0 + 128*x3), tmp16, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-63) + 2*x0 + 128*x3), tmp23, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 128*x3), tmp30, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 128*x3), tmp33, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x3), tmp36, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (63 + 2*x0 + 128*x3), tmp43, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x3), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x3), tmp49, eviction_policy='evict_last', other=float("-inf"))
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


# kernel path: inductor_cache/l4/cl4l5pni6izxnxjnfuzhklctkhwjz2zbph3obm3t7y2bdugdym3r.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 4)
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


# kernel path: inductor_cache/n2/cn2nsewmwfnqb7yqylrnwyqowhdkz7isgvmnxkpgvvuhsksmgzbh.py
# Topologically Sorted Source Nodes: [conv2d_3, out_1, input_11, out_2, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_3
#   input_11 => add_9, mul_13, mul_14, sub_4
#   input_12 => relu_3
#   out_1 => add_7, mul_10, mul_11, sub_3
#   out_2 => add_10
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_17, %primals_18, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %add_9), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 8)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), None)
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp6
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp9 / tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/zo/czoh5op7u5kmukqby2ythxj22thil4e5iqkrf2u3db5wpre7w4ql.py
# Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_41 => add_39, mul_52, mul_53, sub_17
#   input_42 => relu_13
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_39,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 8)
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


# kernel path: inductor_cache/yr/cyrrp32ehzrevlv3f3lqhepi7fahe2lgtcvj57y5w33bfkecne2q.py
# Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_44 => add_41, mul_55, mul_56, sub_18
#   input_45 => relu_14
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 8)
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


# kernel path: inductor_cache/oi/coi4672fn5mrvc54zu4mpfp4zeqgrubg3evynvksnqpmfbfgyogd.py
# Topologically Sorted Source Nodes: [conv2d_19, out_9, input_47, out_10, input_48], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   conv2d_19 => convolution_19
#   input_47 => add_45, mul_61, mul_62, sub_20
#   input_48 => relu_15
#   out_10 => add_46
#   out_9 => add_43, mul_58, mul_59, sub_19
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %primals_101, %primals_102, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_157), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_159), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %add_45), kwargs = {})
#   %relu_15 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_46,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), None)
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp6
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp9 / tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/zm/czmwmtltzy4k6kh7omqydgwy2y32zcwkilfgw3v6z2zxbgue556z.py
# Topologically Sorted Source Nodes: [input_77, input_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_77 => add_75, mul_100, mul_101, sub_33
#   input_78 => relu_25
# Graph fragment:
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_75,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 16)
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


# kernel path: inductor_cache/6n/c6nthjtppm6rzzmnvre2odw7nnqzq4oje37d3rbpsn4fjnez3gxx.py
# Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_80 => add_77, mul_103, mul_104, sub_34
#   input_81 => relu_26
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_273), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_277), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_279), kwargs = {})
#   %relu_26 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 16)
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


# kernel path: inductor_cache/6s/c6sufhm2bwjbgdyaqhfuoyxygkiduzldbhhrlay4tluibeuelalc.py
# Topologically Sorted Source Nodes: [conv2d_35, out_17, input_83, out_18, input_84], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   conv2d_35 => convolution_35
#   input_83 => add_81, mul_109, mul_110, sub_36
#   input_84 => relu_27
#   out_17 => add_79, mul_106, mul_107, sub_35
#   out_18 => add_82
# Graph fragment:
#   %convolution_35 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_26, %primals_185, %primals_186, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_281), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %unsqueeze_285), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_107, %unsqueeze_287), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_289), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_293), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_295), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_79, %add_81), kwargs = {})
#   %relu_27 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_82,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), None)
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp6
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp9 / tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/75/c75gzpkhcjzcmn25dr6fhigwgxcvqcuu6op3y4fv7kieozplki7v.py
# Topologically Sorted Source Nodes: [input_113, input_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_113 => add_111, mul_148, mul_149, sub_49
#   input_114 => relu_37
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_393), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_397), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_399), kwargs = {})
#   %relu_37 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_111,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 32)
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


# kernel path: inductor_cache/jq/cjq6k3m57finaaqroptsakbchvb3wagvpip742n6jgwt54zf73mr.py
# Topologically Sorted Source Nodes: [input_116, input_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_116 => add_113, mul_151, mul_152, sub_50
#   input_117 => relu_38
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_401), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_403), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_151, %unsqueeze_405), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_152, %unsqueeze_407), kwargs = {})
#   %relu_38 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_113,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uq/cuq4zlvdqhs6t4flgjviqac2h2kkr5ag234uwylibo7op2lqx3xk.py
# Topologically Sorted Source Nodes: [conv2d_51, out_25, input_119, out_26, input_120], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   conv2d_51 => convolution_51
#   input_119 => add_117, mul_157, mul_158, sub_52
#   input_120 => relu_39
#   out_25 => add_115, mul_154, mul_155, sub_51
#   out_26 => add_118
# Graph fragment:
#   %convolution_51 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_38, %primals_269, %primals_270, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_409), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, %unsqueeze_413), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %unsqueeze_415), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_417), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %unsqueeze_421), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %unsqueeze_423), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_115, %add_117), kwargs = {})
#   %relu_39 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_118,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x3), None)
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp6
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp9 / tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(in_out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/qy/cqysvunu2gkgb2k5yox4knuxxjuqrovcgn4f5tfldn5sajg5cked.py
# Topologically Sorted Source Nodes: [conv2d_63, out_31, input_146, out_32, input_147, out_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   conv2d_63 => convolution_63
#   input_146 => add_144, mul_193, mul_194, sub_64
#   input_147 => relu_48
#   out_31 => add_142, mul_190, mul_191, sub_63
#   out_32 => add_145
#   out_33 => mean
# Graph fragment:
#   %convolution_63 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_47, %primals_332, %primals_333, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_142 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_513), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %unsqueeze_517), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %unsqueeze_519), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_142, %add_144), kwargs = {})
#   %relu_48 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_145,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_48, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_13 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_13(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 16*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (r2 + 16*x3), xmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp6
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp9 / tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp32 = tl.full([1, 1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp36 = tl.where(xmask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None]
    tmp38 = 16.0
    tmp39 = tmp37 / tmp38
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp39, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, ), (1, ))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (8, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_18, (8, ), (1, ))
    assert_size_stride(primals_19, (8, ), (1, ))
    assert_size_stride(primals_20, (8, ), (1, ))
    assert_size_stride(primals_21, (8, ), (1, ))
    assert_size_stride(primals_22, (8, ), (1, ))
    assert_size_stride(primals_23, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_24, (8, ), (1, ))
    assert_size_stride(primals_25, (8, ), (1, ))
    assert_size_stride(primals_26, (8, ), (1, ))
    assert_size_stride(primals_27, (8, ), (1, ))
    assert_size_stride(primals_28, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, ), (1, ))
    assert_size_stride(primals_33, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, ), (1, ))
    assert_size_stride(primals_38, (8, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_39, (8, ), (1, ))
    assert_size_stride(primals_40, (8, ), (1, ))
    assert_size_stride(primals_41, (8, ), (1, ))
    assert_size_stride(primals_42, (8, ), (1, ))
    assert_size_stride(primals_43, (8, ), (1, ))
    assert_size_stride(primals_44, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_45, (8, ), (1, ))
    assert_size_stride(primals_46, (8, ), (1, ))
    assert_size_stride(primals_47, (8, ), (1, ))
    assert_size_stride(primals_48, (8, ), (1, ))
    assert_size_stride(primals_49, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_50, (4, ), (1, ))
    assert_size_stride(primals_51, (4, ), (1, ))
    assert_size_stride(primals_52, (4, ), (1, ))
    assert_size_stride(primals_53, (4, ), (1, ))
    assert_size_stride(primals_54, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_55, (4, ), (1, ))
    assert_size_stride(primals_56, (4, ), (1, ))
    assert_size_stride(primals_57, (4, ), (1, ))
    assert_size_stride(primals_58, (4, ), (1, ))
    assert_size_stride(primals_59, (8, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_60, (8, ), (1, ))
    assert_size_stride(primals_61, (8, ), (1, ))
    assert_size_stride(primals_62, (8, ), (1, ))
    assert_size_stride(primals_63, (8, ), (1, ))
    assert_size_stride(primals_64, (8, ), (1, ))
    assert_size_stride(primals_65, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_66, (8, ), (1, ))
    assert_size_stride(primals_67, (8, ), (1, ))
    assert_size_stride(primals_68, (8, ), (1, ))
    assert_size_stride(primals_69, (8, ), (1, ))
    assert_size_stride(primals_70, (4, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_71, (4, ), (1, ))
    assert_size_stride(primals_72, (4, ), (1, ))
    assert_size_stride(primals_73, (4, ), (1, ))
    assert_size_stride(primals_74, (4, ), (1, ))
    assert_size_stride(primals_75, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_76, (4, ), (1, ))
    assert_size_stride(primals_77, (4, ), (1, ))
    assert_size_stride(primals_78, (4, ), (1, ))
    assert_size_stride(primals_79, (4, ), (1, ))
    assert_size_stride(primals_80, (8, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_81, (8, ), (1, ))
    assert_size_stride(primals_82, (8, ), (1, ))
    assert_size_stride(primals_83, (8, ), (1, ))
    assert_size_stride(primals_84, (8, ), (1, ))
    assert_size_stride(primals_85, (8, ), (1, ))
    assert_size_stride(primals_86, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_87, (8, ), (1, ))
    assert_size_stride(primals_88, (8, ), (1, ))
    assert_size_stride(primals_89, (8, ), (1, ))
    assert_size_stride(primals_90, (8, ), (1, ))
    assert_size_stride(primals_91, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_92, (8, ), (1, ))
    assert_size_stride(primals_93, (8, ), (1, ))
    assert_size_stride(primals_94, (8, ), (1, ))
    assert_size_stride(primals_95, (8, ), (1, ))
    assert_size_stride(primals_96, (8, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_97, (8, ), (1, ))
    assert_size_stride(primals_98, (8, ), (1, ))
    assert_size_stride(primals_99, (8, ), (1, ))
    assert_size_stride(primals_100, (8, ), (1, ))
    assert_size_stride(primals_101, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_102, (16, ), (1, ))
    assert_size_stride(primals_103, (16, ), (1, ))
    assert_size_stride(primals_104, (16, ), (1, ))
    assert_size_stride(primals_105, (16, ), (1, ))
    assert_size_stride(primals_106, (16, ), (1, ))
    assert_size_stride(primals_107, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_108, (16, ), (1, ))
    assert_size_stride(primals_109, (16, ), (1, ))
    assert_size_stride(primals_110, (16, ), (1, ))
    assert_size_stride(primals_111, (16, ), (1, ))
    assert_size_stride(primals_112, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_113, (8, ), (1, ))
    assert_size_stride(primals_114, (8, ), (1, ))
    assert_size_stride(primals_115, (8, ), (1, ))
    assert_size_stride(primals_116, (8, ), (1, ))
    assert_size_stride(primals_117, (8, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_118, (8, ), (1, ))
    assert_size_stride(primals_119, (8, ), (1, ))
    assert_size_stride(primals_120, (8, ), (1, ))
    assert_size_stride(primals_121, (8, ), (1, ))
    assert_size_stride(primals_122, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_123, (16, ), (1, ))
    assert_size_stride(primals_124, (16, ), (1, ))
    assert_size_stride(primals_125, (16, ), (1, ))
    assert_size_stride(primals_126, (16, ), (1, ))
    assert_size_stride(primals_127, (16, ), (1, ))
    assert_size_stride(primals_128, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_129, (16, ), (1, ))
    assert_size_stride(primals_130, (16, ), (1, ))
    assert_size_stride(primals_131, (16, ), (1, ))
    assert_size_stride(primals_132, (16, ), (1, ))
    assert_size_stride(primals_133, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_134, (8, ), (1, ))
    assert_size_stride(primals_135, (8, ), (1, ))
    assert_size_stride(primals_136, (8, ), (1, ))
    assert_size_stride(primals_137, (8, ), (1, ))
    assert_size_stride(primals_138, (8, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_139, (8, ), (1, ))
    assert_size_stride(primals_140, (8, ), (1, ))
    assert_size_stride(primals_141, (8, ), (1, ))
    assert_size_stride(primals_142, (8, ), (1, ))
    assert_size_stride(primals_143, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_144, (16, ), (1, ))
    assert_size_stride(primals_145, (16, ), (1, ))
    assert_size_stride(primals_146, (16, ), (1, ))
    assert_size_stride(primals_147, (16, ), (1, ))
    assert_size_stride(primals_148, (16, ), (1, ))
    assert_size_stride(primals_149, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_150, (16, ), (1, ))
    assert_size_stride(primals_151, (16, ), (1, ))
    assert_size_stride(primals_152, (16, ), (1, ))
    assert_size_stride(primals_153, (16, ), (1, ))
    assert_size_stride(primals_154, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_155, (8, ), (1, ))
    assert_size_stride(primals_156, (8, ), (1, ))
    assert_size_stride(primals_157, (8, ), (1, ))
    assert_size_stride(primals_158, (8, ), (1, ))
    assert_size_stride(primals_159, (8, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_160, (8, ), (1, ))
    assert_size_stride(primals_161, (8, ), (1, ))
    assert_size_stride(primals_162, (8, ), (1, ))
    assert_size_stride(primals_163, (8, ), (1, ))
    assert_size_stride(primals_164, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_165, (16, ), (1, ))
    assert_size_stride(primals_166, (16, ), (1, ))
    assert_size_stride(primals_167, (16, ), (1, ))
    assert_size_stride(primals_168, (16, ), (1, ))
    assert_size_stride(primals_169, (16, ), (1, ))
    assert_size_stride(primals_170, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_171, (16, ), (1, ))
    assert_size_stride(primals_172, (16, ), (1, ))
    assert_size_stride(primals_173, (16, ), (1, ))
    assert_size_stride(primals_174, (16, ), (1, ))
    assert_size_stride(primals_175, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_176, (16, ), (1, ))
    assert_size_stride(primals_177, (16, ), (1, ))
    assert_size_stride(primals_178, (16, ), (1, ))
    assert_size_stride(primals_179, (16, ), (1, ))
    assert_size_stride(primals_180, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_181, (16, ), (1, ))
    assert_size_stride(primals_182, (16, ), (1, ))
    assert_size_stride(primals_183, (16, ), (1, ))
    assert_size_stride(primals_184, (16, ), (1, ))
    assert_size_stride(primals_185, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_186, (32, ), (1, ))
    assert_size_stride(primals_187, (32, ), (1, ))
    assert_size_stride(primals_188, (32, ), (1, ))
    assert_size_stride(primals_189, (32, ), (1, ))
    assert_size_stride(primals_190, (32, ), (1, ))
    assert_size_stride(primals_191, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_192, (32, ), (1, ))
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_194, (32, ), (1, ))
    assert_size_stride(primals_195, (32, ), (1, ))
    assert_size_stride(primals_196, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_197, (16, ), (1, ))
    assert_size_stride(primals_198, (16, ), (1, ))
    assert_size_stride(primals_199, (16, ), (1, ))
    assert_size_stride(primals_200, (16, ), (1, ))
    assert_size_stride(primals_201, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_202, (16, ), (1, ))
    assert_size_stride(primals_203, (16, ), (1, ))
    assert_size_stride(primals_204, (16, ), (1, ))
    assert_size_stride(primals_205, (16, ), (1, ))
    assert_size_stride(primals_206, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_207, (32, ), (1, ))
    assert_size_stride(primals_208, (32, ), (1, ))
    assert_size_stride(primals_209, (32, ), (1, ))
    assert_size_stride(primals_210, (32, ), (1, ))
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_212, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_213, (32, ), (1, ))
    assert_size_stride(primals_214, (32, ), (1, ))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, ), (1, ))
    assert_size_stride(primals_217, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_218, (16, ), (1, ))
    assert_size_stride(primals_219, (16, ), (1, ))
    assert_size_stride(primals_220, (16, ), (1, ))
    assert_size_stride(primals_221, (16, ), (1, ))
    assert_size_stride(primals_222, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_223, (16, ), (1, ))
    assert_size_stride(primals_224, (16, ), (1, ))
    assert_size_stride(primals_225, (16, ), (1, ))
    assert_size_stride(primals_226, (16, ), (1, ))
    assert_size_stride(primals_227, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_228, (32, ), (1, ))
    assert_size_stride(primals_229, (32, ), (1, ))
    assert_size_stride(primals_230, (32, ), (1, ))
    assert_size_stride(primals_231, (32, ), (1, ))
    assert_size_stride(primals_232, (32, ), (1, ))
    assert_size_stride(primals_233, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_234, (32, ), (1, ))
    assert_size_stride(primals_235, (32, ), (1, ))
    assert_size_stride(primals_236, (32, ), (1, ))
    assert_size_stride(primals_237, (32, ), (1, ))
    assert_size_stride(primals_238, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_239, (16, ), (1, ))
    assert_size_stride(primals_240, (16, ), (1, ))
    assert_size_stride(primals_241, (16, ), (1, ))
    assert_size_stride(primals_242, (16, ), (1, ))
    assert_size_stride(primals_243, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_244, (16, ), (1, ))
    assert_size_stride(primals_245, (16, ), (1, ))
    assert_size_stride(primals_246, (16, ), (1, ))
    assert_size_stride(primals_247, (16, ), (1, ))
    assert_size_stride(primals_248, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_249, (32, ), (1, ))
    assert_size_stride(primals_250, (32, ), (1, ))
    assert_size_stride(primals_251, (32, ), (1, ))
    assert_size_stride(primals_252, (32, ), (1, ))
    assert_size_stride(primals_253, (32, ), (1, ))
    assert_size_stride(primals_254, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_255, (32, ), (1, ))
    assert_size_stride(primals_256, (32, ), (1, ))
    assert_size_stride(primals_257, (32, ), (1, ))
    assert_size_stride(primals_258, (32, ), (1, ))
    assert_size_stride(primals_259, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_260, (32, ), (1, ))
    assert_size_stride(primals_261, (32, ), (1, ))
    assert_size_stride(primals_262, (32, ), (1, ))
    assert_size_stride(primals_263, (32, ), (1, ))
    assert_size_stride(primals_264, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, ), (1, ))
    assert_size_stride(primals_267, (32, ), (1, ))
    assert_size_stride(primals_268, (32, ), (1, ))
    assert_size_stride(primals_269, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_270, (64, ), (1, ))
    assert_size_stride(primals_271, (64, ), (1, ))
    assert_size_stride(primals_272, (64, ), (1, ))
    assert_size_stride(primals_273, (64, ), (1, ))
    assert_size_stride(primals_274, (64, ), (1, ))
    assert_size_stride(primals_275, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_276, (64, ), (1, ))
    assert_size_stride(primals_277, (64, ), (1, ))
    assert_size_stride(primals_278, (64, ), (1, ))
    assert_size_stride(primals_279, (64, ), (1, ))
    assert_size_stride(primals_280, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_281, (32, ), (1, ))
    assert_size_stride(primals_282, (32, ), (1, ))
    assert_size_stride(primals_283, (32, ), (1, ))
    assert_size_stride(primals_284, (32, ), (1, ))
    assert_size_stride(primals_285, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_286, (32, ), (1, ))
    assert_size_stride(primals_287, (32, ), (1, ))
    assert_size_stride(primals_288, (32, ), (1, ))
    assert_size_stride(primals_289, (32, ), (1, ))
    assert_size_stride(primals_290, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_291, (64, ), (1, ))
    assert_size_stride(primals_292, (64, ), (1, ))
    assert_size_stride(primals_293, (64, ), (1, ))
    assert_size_stride(primals_294, (64, ), (1, ))
    assert_size_stride(primals_295, (64, ), (1, ))
    assert_size_stride(primals_296, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_297, (64, ), (1, ))
    assert_size_stride(primals_298, (64, ), (1, ))
    assert_size_stride(primals_299, (64, ), (1, ))
    assert_size_stride(primals_300, (64, ), (1, ))
    assert_size_stride(primals_301, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_302, (32, ), (1, ))
    assert_size_stride(primals_303, (32, ), (1, ))
    assert_size_stride(primals_304, (32, ), (1, ))
    assert_size_stride(primals_305, (32, ), (1, ))
    assert_size_stride(primals_306, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_307, (32, ), (1, ))
    assert_size_stride(primals_308, (32, ), (1, ))
    assert_size_stride(primals_309, (32, ), (1, ))
    assert_size_stride(primals_310, (32, ), (1, ))
    assert_size_stride(primals_311, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_312, (64, ), (1, ))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (64, ), (1, ))
    assert_size_stride(primals_315, (64, ), (1, ))
    assert_size_stride(primals_316, (64, ), (1, ))
    assert_size_stride(primals_317, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (64, ), (1, ))
    assert_size_stride(primals_321, (64, ), (1, ))
    assert_size_stride(primals_322, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_323, (32, ), (1, ))
    assert_size_stride(primals_324, (32, ), (1, ))
    assert_size_stride(primals_325, (32, ), (1, ))
    assert_size_stride(primals_326, (32, ), (1, ))
    assert_size_stride(primals_327, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_328, (32, ), (1, ))
    assert_size_stride(primals_329, (32, ), (1, ))
    assert_size_stride(primals_330, (32, ), (1, ))
    assert_size_stride(primals_331, (32, ), (1, ))
    assert_size_stride(primals_332, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_333, (64, ), (1, ))
    assert_size_stride(primals_334, (64, ), (1, ))
    assert_size_stride(primals_335, (64, ), (1, ))
    assert_size_stride(primals_336, (64, ), (1, ))
    assert_size_stride(primals_337, (64, ), (1, ))
    assert_size_stride(primals_338, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (64, ), (1, ))
    assert_size_stride(primals_341, (64, ), (1, ))
    assert_size_stride(primals_342, (64, ), (1, ))
    assert_size_stride(primals_343, (4, 64), (64, 1))
    assert_size_stride(primals_344, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf1 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 1048576, grid=grid(1048576), stream=stream0)
        del primals_6
        buf2 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.int8)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(buf1, buf2, buf3, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf5 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, primals_8, primals_9, primals_10, primals_11, buf5, 16384, grid=grid(16384), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf6, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf7 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf6, primals_13, primals_14, primals_15, primals_16, buf7, 16384, grid=grid(16384), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf2, primals_23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf9 = buf8; del buf8  # reuse
        buf11 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [conv2d_3, out_1, input_11, out_2, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf9, buf12, primals_18, primals_19, primals_20, primals_21, primals_22, buf10, primals_24, primals_25, primals_26, primals_27, 32768, grid=grid(32768), stream=stream0)
        del primals_18
        del primals_22
        del primals_27
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf14 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf13, primals_29, primals_30, primals_31, primals_32, buf14, 16384, grid=grid(16384), stream=stream0)
        del primals_32
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf15, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf16 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf15, primals_34, primals_35, primals_36, primals_37, buf16, 16384, grid=grid(16384), stream=stream0)
        del primals_37
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf12, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf18 = buf17; del buf17  # reuse
        buf20 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [conv2d_7, out_3, input_20, out_4, input_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf18, buf21, primals_39, primals_40, primals_41, primals_42, primals_43, buf19, primals_45, primals_46, primals_47, primals_48, 32768, grid=grid(32768), stream=stream0)
        del primals_39
        del primals_43
        del primals_48
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf23 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf22, primals_50, primals_51, primals_52, primals_53, buf23, 16384, grid=grid(16384), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf24, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf25 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf24, primals_55, primals_56, primals_57, primals_58, buf25, 16384, grid=grid(16384), stream=stream0)
        del primals_58
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_59, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf21, primals_65, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf27 = buf26; del buf26  # reuse
        buf29 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [conv2d_11, out_5, input_29, out_6, input_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf27, buf30, primals_60, primals_61, primals_62, primals_63, primals_64, buf28, primals_66, primals_67, primals_68, primals_69, 32768, grid=grid(32768), stream=stream0)
        del primals_60
        del primals_64
        del primals_69
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf32 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf31, primals_71, primals_72, primals_73, primals_74, buf32, 16384, grid=grid(16384), stream=stream0)
        del primals_74
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf33, (4, 4, 32, 32), (4096, 1024, 32, 1))
        buf34 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf33, primals_76, primals_77, primals_78, primals_79, buf34, 16384, grid=grid(16384), stream=stream0)
        del primals_79
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 8, 32, 32), (8192, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf30, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf36 = buf35; del buf35  # reuse
        buf38 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [conv2d_15, out_7, input_38, out_8, input_39], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_3.run(buf36, buf39, primals_81, primals_82, primals_83, primals_84, primals_85, buf37, primals_87, primals_88, primals_89, primals_90, 32768, grid=grid(32768), stream=stream0)
        del primals_81
        del primals_85
        del primals_90
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 8, 32, 32), (8192, 1024, 32, 1))
        buf41 = empty_strided_cuda((4, 8, 32, 32), (8192, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf40, primals_92, primals_93, primals_94, primals_95, buf41, 32768, grid=grid(32768), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_96, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf42, (4, 8, 16, 16), (2048, 256, 16, 1))
        buf43 = empty_strided_cuda((4, 8, 16, 16), (2048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf42, primals_97, primals_98, primals_99, primals_100, buf43, 8192, grid=grid(8192), stream=stream0)
        del primals_100
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf39, primals_107, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf45 = buf44; del buf44  # reuse
        buf47 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [conv2d_19, out_9, input_47, out_10, input_48], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6.run(buf45, buf48, primals_102, primals_103, primals_104, primals_105, primals_106, buf46, primals_108, primals_109, primals_110, primals_111, 16384, grid=grid(16384), stream=stream0)
        del primals_102
        del primals_106
        del primals_111
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 8, 16, 16), (2048, 256, 16, 1))
        buf50 = empty_strided_cuda((4, 8, 16, 16), (2048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, input_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf49, primals_113, primals_114, primals_115, primals_116, buf50, 8192, grid=grid(8192), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf51, (4, 8, 16, 16), (2048, 256, 16, 1))
        buf52 = empty_strided_cuda((4, 8, 16, 16), (2048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_53, input_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf51, primals_118, primals_119, primals_120, primals_121, buf52, 8192, grid=grid(8192), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf48, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf54 = buf53; del buf53  # reuse
        buf56 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [conv2d_23, out_11, input_56, out_12, input_57], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6.run(buf54, buf57, primals_123, primals_124, primals_125, primals_126, primals_127, buf55, primals_129, primals_130, primals_131, primals_132, 16384, grid=grid(16384), stream=stream0)
        del primals_123
        del primals_127
        del primals_132
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 8, 16, 16), (2048, 256, 16, 1))
        buf59 = empty_strided_cuda((4, 8, 16, 16), (2048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_59, input_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf58, primals_134, primals_135, primals_136, primals_137, buf59, 8192, grid=grid(8192), stream=stream0)
        del primals_137
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_138, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf60, (4, 8, 16, 16), (2048, 256, 16, 1))
        buf61 = empty_strided_cuda((4, 8, 16, 16), (2048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf60, primals_139, primals_140, primals_141, primals_142, buf61, 8192, grid=grid(8192), stream=stream0)
        del primals_142
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf57, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf63 = buf62; del buf62  # reuse
        buf65 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [conv2d_27, out_13, input_65, out_14, input_66], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6.run(buf63, buf66, primals_144, primals_145, primals_146, primals_147, primals_148, buf64, primals_150, primals_151, primals_152, primals_153, 16384, grid=grid(16384), stream=stream0)
        del primals_144
        del primals_148
        del primals_153
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 8, 16, 16), (2048, 256, 16, 1))
        buf68 = empty_strided_cuda((4, 8, 16, 16), (2048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf67, primals_155, primals_156, primals_157, primals_158, buf68, 8192, grid=grid(8192), stream=stream0)
        del primals_158
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf69, (4, 8, 16, 16), (2048, 256, 16, 1))
        buf70 = empty_strided_cuda((4, 8, 16, 16), (2048, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf69, primals_160, primals_161, primals_162, primals_163, buf70, 8192, grid=grid(8192), stream=stream0)
        del primals_163
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 16, 16, 16), (4096, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf66, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf72 = buf71; del buf71  # reuse
        buf74 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [conv2d_31, out_15, input_74, out_16, input_75], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6.run(buf72, buf75, primals_165, primals_166, primals_167, primals_168, primals_169, buf73, primals_171, primals_172, primals_173, primals_174, 16384, grid=grid(16384), stream=stream0)
        del primals_165
        del primals_169
        del primals_174
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 16, 16, 16), (4096, 256, 16, 1))
        buf77 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_77, input_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf76, primals_176, primals_177, primals_178, primals_179, buf77, 16384, grid=grid(16384), stream=stream0)
        del primals_179
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_180, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf78, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf79 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf78, primals_181, primals_182, primals_183, primals_184, buf79, 4096, grid=grid(4096), stream=stream0)
        del primals_184
        # Topologically Sorted Source Nodes: [conv2d_35], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf75, primals_191, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf81 = buf80; del buf80  # reuse
        buf83 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [conv2d_35, out_17, input_83, out_18, input_84], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9.run(buf81, buf84, primals_186, primals_187, primals_188, primals_189, primals_190, buf82, primals_192, primals_193, primals_194, primals_195, 8192, grid=grid(8192), stream=stream0)
        del primals_186
        del primals_190
        del primals_195
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf86 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_86, input_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf85, primals_197, primals_198, primals_199, primals_200, buf86, 4096, grid=grid(4096), stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_201, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf87, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf88 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_89, input_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf87, primals_202, primals_203, primals_204, primals_205, buf88, 4096, grid=grid(4096), stream=stream0)
        del primals_205
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf84, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf90 = buf89; del buf89  # reuse
        buf92 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [conv2d_39, out_19, input_92, out_20, input_93], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9.run(buf90, buf93, primals_207, primals_208, primals_209, primals_210, primals_211, buf91, primals_213, primals_214, primals_215, primals_216, 8192, grid=grid(8192), stream=stream0)
        del primals_207
        del primals_211
        del primals_216
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf95 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_95, input_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf94, primals_218, primals_219, primals_220, primals_221, buf95, 4096, grid=grid(4096), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [input_97], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf96, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf97 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_98, input_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf96, primals_223, primals_224, primals_225, primals_226, buf97, 4096, grid=grid(4096), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf93, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf99 = buf98; del buf98  # reuse
        buf101 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [conv2d_43, out_21, input_101, out_22, input_102], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9.run(buf99, buf102, primals_228, primals_229, primals_230, primals_231, primals_232, buf100, primals_234, primals_235, primals_236, primals_237, 8192, grid=grid(8192), stream=stream0)
        del primals_228
        del primals_232
        del primals_237
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf104 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf103, primals_239, primals_240, primals_241, primals_242, buf104, 4096, grid=grid(4096), stream=stream0)
        del primals_242
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_243, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf105, (4, 16, 8, 8), (1024, 64, 8, 1))
        buf106 = empty_strided_cuda((4, 16, 8, 8), (1024, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_107, input_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf105, primals_244, primals_245, primals_246, primals_247, buf106, 4096, grid=grid(4096), stream=stream0)
        del primals_247
        # Topologically Sorted Source Nodes: [conv2d_47], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 32, 8, 8), (2048, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf102, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf108 = buf107; del buf107  # reuse
        buf110 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [conv2d_47, out_23, input_110, out_24, input_111], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_9.run(buf108, buf111, primals_249, primals_250, primals_251, primals_252, primals_253, buf109, primals_255, primals_256, primals_257, primals_258, 8192, grid=grid(8192), stream=stream0)
        del primals_249
        del primals_253
        del primals_258
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 32, 8, 8), (2048, 64, 8, 1))
        buf113 = empty_strided_cuda((4, 32, 8, 8), (2048, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_113, input_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf112, primals_260, primals_261, primals_262, primals_263, buf113, 8192, grid=grid(8192), stream=stream0)
        del primals_263
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_264, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf114, (4, 32, 4, 4), (512, 16, 4, 1))
        buf115 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_116, input_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf114, primals_265, primals_266, primals_267, primals_268, buf115, 2048, grid=grid(2048), stream=stream0)
        del primals_268
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_118], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf111, primals_275, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf117 = buf116; del buf116  # reuse
        buf119 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf120 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [conv2d_51, out_25, input_119, out_26, input_120], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12.run(buf117, buf120, primals_270, primals_271, primals_272, primals_273, primals_274, buf118, primals_276, primals_277, primals_278, primals_279, 4096, grid=grid(4096), stream=stream0)
        del primals_270
        del primals_274
        del primals_279
        # Topologically Sorted Source Nodes: [input_121], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 32, 4, 4), (512, 16, 4, 1))
        buf122 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_122, input_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf121, primals_281, primals_282, primals_283, primals_284, buf122, 2048, grid=grid(2048), stream=stream0)
        del primals_284
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf123, (4, 32, 4, 4), (512, 16, 4, 1))
        buf124 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_125, input_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf123, primals_286, primals_287, primals_288, primals_289, buf124, 2048, grid=grid(2048), stream=stream0)
        del primals_289
        # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_290, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_127], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf120, primals_296, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf126 = buf125; del buf125  # reuse
        buf128 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf129 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [conv2d_55, out_27, input_128, out_28, input_129], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12.run(buf126, buf129, primals_291, primals_292, primals_293, primals_294, primals_295, buf127, primals_297, primals_298, primals_299, primals_300, 4096, grid=grid(4096), stream=stream0)
        del primals_291
        del primals_295
        del primals_300
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 32, 4, 4), (512, 16, 4, 1))
        buf131 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_131, input_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf130, primals_302, primals_303, primals_304, primals_305, buf131, 2048, grid=grid(2048), stream=stream0)
        del primals_305
        # Topologically Sorted Source Nodes: [input_133], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf132, (4, 32, 4, 4), (512, 16, 4, 1))
        buf133 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_134, input_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf132, primals_307, primals_308, primals_309, primals_310, buf133, 2048, grid=grid(2048), stream=stream0)
        del primals_310
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf129, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf135 = buf134; del buf134  # reuse
        buf137 = empty_strided_cuda((4, 64, 4, 4), (1024, 16, 4, 1), torch.float32)
        buf138 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [conv2d_59, out_29, input_137, out_30, input_138], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_12.run(buf135, buf138, primals_312, primals_313, primals_314, primals_315, primals_316, buf136, primals_318, primals_319, primals_320, primals_321, 4096, grid=grid(4096), stream=stream0)
        del primals_312
        del primals_316
        del primals_321
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 32, 4, 4), (512, 16, 4, 1))
        buf140 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_140, input_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf139, primals_323, primals_324, primals_325, primals_326, buf140, 2048, grid=grid(2048), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [input_142], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf141, (4, 32, 4, 4), (512, 16, 4, 1))
        buf142 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_143, input_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf141, primals_328, primals_329, primals_330, primals_331, buf142, 2048, grid=grid(2048), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 64, 4, 4), (1024, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_145], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf138, primals_338, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 64, 4, 4), (1024, 16, 4, 1))
        buf144 = buf143; del buf143  # reuse
        buf147 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [conv2d_63, out_31, input_146, out_32, input_147, out_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_13.run(buf144, buf148, primals_333, primals_334, primals_335, primals_336, primals_337, buf145, primals_339, primals_340, primals_341, primals_342, 256, 16, grid=grid(256), stream=stream0)
        del primals_333
        buf149 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_344, reinterpret_tensor(buf148, (4, 64), (64, 1), 0), reinterpret_tensor(primals_343, (64, 4), (1, 64), 0), alpha=1, beta=1, out=buf149)
        del primals_344
    return (buf149, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_54, primals_55, primals_56, primals_57, primals_59, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_70, primals_71, primals_72, primals_73, primals_75, primals_76, primals_77, primals_78, primals_80, primals_82, primals_83, primals_84, primals_86, primals_87, primals_88, primals_89, primals_91, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_101, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_128, primals_129, primals_130, primals_131, primals_133, primals_134, primals_135, primals_136, primals_138, primals_139, primals_140, primals_141, primals_143, primals_145, primals_146, primals_147, primals_149, primals_150, primals_151, primals_152, primals_154, primals_155, primals_156, primals_157, primals_159, primals_160, primals_161, primals_162, primals_164, primals_166, primals_167, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_176, primals_177, primals_178, primals_180, primals_181, primals_182, primals_183, primals_185, primals_187, primals_188, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_202, primals_203, primals_204, primals_206, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_227, primals_229, primals_230, primals_231, primals_233, primals_234, primals_235, primals_236, primals_238, primals_239, primals_240, primals_241, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_254, primals_255, primals_256, primals_257, primals_259, primals_260, primals_261, primals_262, primals_264, primals_265, primals_266, primals_267, primals_269, primals_271, primals_272, primals_273, primals_275, primals_276, primals_277, primals_278, primals_280, primals_281, primals_282, primals_283, primals_285, primals_286, primals_287, primals_288, primals_290, primals_292, primals_293, primals_294, primals_296, primals_297, primals_298, primals_299, primals_301, primals_302, primals_303, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_313, primals_314, primals_315, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf9, buf10, buf12, buf13, buf14, buf15, buf16, buf18, buf19, buf21, buf22, buf23, buf24, buf25, buf27, buf28, buf30, buf31, buf32, buf33, buf34, buf36, buf37, buf39, buf40, buf41, buf42, buf43, buf45, buf46, buf48, buf49, buf50, buf51, buf52, buf54, buf55, buf57, buf58, buf59, buf60, buf61, buf63, buf64, buf66, buf67, buf68, buf69, buf70, buf72, buf73, buf75, buf76, buf77, buf78, buf79, buf81, buf82, buf84, buf85, buf86, buf87, buf88, buf90, buf91, buf93, buf94, buf95, buf96, buf97, buf99, buf100, buf102, buf103, buf104, buf105, buf106, buf108, buf109, buf111, buf112, buf113, buf114, buf115, buf117, buf118, buf120, buf121, buf122, buf123, buf124, buf126, buf127, buf129, buf130, buf131, buf132, buf133, buf135, buf136, buf138, buf139, buf140, buf141, buf142, buf144, buf145, reinterpret_tensor(buf148, (4, 64), (64, 1), 0), primals_343, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((8, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((8, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((4, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((8, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((8, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((8, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((8, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((8, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((32, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((32, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((32, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((32, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((4, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
