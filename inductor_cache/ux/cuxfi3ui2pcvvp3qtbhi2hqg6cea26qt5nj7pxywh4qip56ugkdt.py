# AOT ID: ['13_forward']
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


# kernel path: inductor_cache/5m/c5maepkiwsp7xrhsjzhibh6owtkoxcgtmbadyxuws6jxamdhttoq.py
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
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
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


# kernel path: inductor_cache/r6/cr6dhvqdofz5ckjxqx64m7zxdldqqmgduo7tpl77o5ufbdidf6qc.py
# Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_4 => _low_memory_max_pool2d_with_offsets, getitem_1
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => relu_1
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %sub_240 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %unsqueeze_2398), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x6 = xindex // 16
    x3 = xindex // 16384
    x7 = (xindex % 16384)
    x8 = xindex
    x2 = ((xindex // 256) % 64)
    tmp77 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + 2*x0 + 64*x6), tmp10, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + 2*x0 + 64*x6), tmp16, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + 2*x0 + 64*x6), tmp23, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x6), tmp30, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 64*x6), tmp33, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x6), tmp36, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + 2*x0 + 64*x6), tmp43, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x6), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x6), tmp49, eviction_policy='evict_last', other=float("-inf"))
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
    tmp78 = tmp51 - tmp77
    tmp80 = 1e-05
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp89 = tmp87 * tmp88
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tl.store(out_ptr0 + (x7 + 19456*x3), tmp51, None)
    tl.store(out_ptr1 + (x8), tmp76, None)
    tl.store(out_ptr2 + (x8), tmp93, None)
    tl.store(out_ptr3 + (x8), tmp78, None)
''', device_str='cuda')


# kernel path: inductor_cache/ow/cowf2d3nou5qdlz7s4cs4nnj2xwk7aqb6buhk7avs7g2eb43y3ko.py
# Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_8 => add_5, mul_7, mul_8, sub_2
#   input_9 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 48)
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


# kernel path: inductor_cache/vn/cvnnxmo7qfscdrae6ihcyt2enleefvqobziqxh5dgpae5gooo6e7.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   input_11 => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %convolution_2], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + 19456*x1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/vy/cvyfivxzrq6stqljq2nao36mncic3c24zxu6ovh2cujcqbgrbaia.py
# Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_12 => add_7, mul_10, mul_11, sub_3
#   input_13 => relu_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 77824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 76)
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


# kernel path: inductor_cache/ob/cobfvdnavtmiwqer77skyqnkdek2mwd4qr36vrcik5gjrdfmtudy.py
# Topologically Sorted Source Nodes: [input_18, input_19, input_20], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_18 => cat_1
#   input_19 => add_11, mul_16, mul_17, sub_5
#   input_20 => relu_5
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat, %convolution_4], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 88)
    x0 = (xindex % 256)
    x2 = xindex // 22528
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 76, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 19456*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 88, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 256*((-76) + x1) + 3072*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/5c/c5ca52rxgk52g62bils6hjoyogmo477dfzflfa4eyr5kmcfbnlfr.py
# Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_25 => cat_2
#   input_26 => add_15, mul_22, mul_23, sub_7
#   input_27 => relu_7
# Graph fragment:
#   %cat_2 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_1, %convolution_6], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_2, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 100)
    x0 = (xindex % 256)
    x2 = xindex // 25600
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 88, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 22528*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 100, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 256*((-88) + x1) + 3072*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/zj/czjji4p5mtlxh73hm2qjqa4ycqk2wbhymqocrextfu5vzucslzxr.py
# Topologically Sorted Source Nodes: [input_32, input_33, input_34], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_32 => cat_3
#   input_33 => add_19, mul_28, mul_29, sub_9
#   input_34 => relu_9
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %convolution_8], 1), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_3, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 114688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 112)
    x0 = (xindex % 256)
    x2 = xindex // 28672
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 100, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 25600*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 112, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 256*((-100) + x1) + 3072*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/dc/cdcqul7kntuyplwsfs4psir574pkxzeztzgfsubzm2bkxwwxt2cj.py
# Topologically Sorted Source Nodes: [input_39, input_40, input_41], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_39 => cat_4
#   input_40 => add_23, mul_34, mul_35, sub_11
#   input_41 => relu_11
# Graph fragment:
#   %cat_4 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_3, %convolution_10], 1), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_4, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 126976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 124)
    x0 = (xindex % 256)
    x2 = xindex // 31744
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 28672*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 124, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 256*((-112) + x1) + 3072*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/4c/c4cxeboxycjm4yigw37zg5rb4dggd3x66ge2r3jiuzgrmhycmztq.py
# Topologically Sorted Source Nodes: [input_46, input_47, input_48], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_46 => cat_5
#   input_47 => add_27, mul_40, mul_41, sub_13
#   input_48 => relu_13
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_4, %convolution_12], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_5, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %relu_13 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_27,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 139264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 136)
    x0 = (xindex % 256)
    x2 = xindex // 34816
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 124, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1) + 31744*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 136, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 256*((-124) + x1) + 3072*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/fg/cfgedqhyfaqtqyiqpvgfbnos6ygj4ii4ol77b6gxb3hcjgdjoydx.py
# Topologically Sorted Source Nodes: [input_50, input_51, input_52], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_50 => avg_pool2d
#   input_51 => add_29, mul_43, mul_44, sub_14
#   input_52 => relu_14
# Graph fragment:
#   %avg_pool2d : [num_users=3] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convolution_13, [2, 2], [2, 2]), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x5 = xindex
    x3 = ((xindex // 64) % 68)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lx/clxy6qjbjitfao6ct55g2hjpsgl3jif4zr37tuuhxixapls44cid.py
# Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_54 => add_31, mul_46, mul_47, sub_15
#   input_55 => relu_15
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 48)
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


# kernel path: inductor_cache/2w/c2wqs27ivo55fpabgshlqgkcd2zswotphgrpo6fnlxizp5nfgvh6.py
# Topologically Sorted Source Nodes: [input_57, input_58, input_59], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_57 => cat_6
#   input_58 => add_33, mul_49, mul_50, sub_16
#   input_59 => relu_16
# Graph fragment:
#   %cat_6 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%avg_pool2d, %convolution_15], 1), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_6, %unsqueeze_129), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %relu_16 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 80)
    x0 = (xindex % 64)
    x2 = xindex // 5120
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 68, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 4352*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 80, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-68) + x1) + 768*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/6i/c6iurk42zqh2fiyfosv2udh5bhayxexd2lqkfta54iglttfiau7m.py
# Topologically Sorted Source Nodes: [input_64, input_65, input_66], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_64 => cat_7
#   input_65 => add_37, mul_55, mul_56, sub_18
#   input_66 => relu_18
# Graph fragment:
#   %cat_7 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_6, %convolution_17], 1), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_7, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 92)
    x0 = (xindex % 64)
    x2 = xindex // 5888
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 5120*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 92, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-80) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3t/c3tzvjicw3xazdorjrrt7c237dqaozq3wqerulaaqupbmqxvv7gy.py
# Topologically Sorted Source Nodes: [input_71, input_72, input_73], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_71 => cat_8
#   input_72 => add_41, mul_61, mul_62, sub_20
#   input_73 => relu_20
# Graph fragment:
#   %cat_8 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_7, %convolution_19], 1), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_8, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %relu_20 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 104)
    x0 = (xindex % 64)
    x2 = xindex // 6656
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 92, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 5888*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 104, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-92) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4g/c4g4aq6dsdtswz52md3knr2hhxil3ycxd526udvd2oq4qvsu6iug.py
# Topologically Sorted Source Nodes: [input_78, input_79, input_80], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_78 => cat_9
#   input_79 => add_45, mul_67, mul_68, sub_22
#   input_80 => relu_22
# Graph fragment:
#   %cat_9 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_8, %convolution_21], 1), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_9, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_45,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 116)
    x0 = (xindex % 64)
    x2 = xindex // 7424
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 104, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 6656*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 116, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-104) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2o/c2ogooyumjfj6ik46keiybotnbbkul6jpkyeb2e2cyyb5k7q55au.py
# Topologically Sorted Source Nodes: [input_85, input_86, input_87], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_85 => cat_10
#   input_86 => add_49, mul_73, mul_74, sub_24
#   input_87 => relu_24
# Graph fragment:
#   %cat_10 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_9, %convolution_23], 1), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_10, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_49,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 128)
    x0 = (xindex % 64)
    x2 = xindex // 8192
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 116, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 7424*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-116) + x1) + 768*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/t7/ct7zns3zwpr3jmnjt2xmfv6mwblekvbbxidltui3mwmijkus7fzn.py
# Topologically Sorted Source Nodes: [input_92, input_93, input_94], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_92 => cat_11
#   input_93 => add_53, mul_79, mul_80, sub_26
#   input_94 => relu_26
# Graph fragment:
#   %cat_11 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_10, %convolution_25], 1), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_11, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 35840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 140)
    x0 = (xindex % 64)
    x2 = xindex // 8960
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8192*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 140, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-128) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vp/cvpz4ngydokckp5zu46k3s6ij4smi2q36crpfze6d3j3g5wvxvii.py
# Topologically Sorted Source Nodes: [input_99, input_100, input_101], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_100 => add_57, mul_85, mul_86, sub_28
#   input_101 => relu_28
#   input_99 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_11, %convolution_27], 1), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_12, %unsqueeze_225), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %relu_28 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_57,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 152)
    x0 = (xindex % 64)
    x2 = xindex // 9728
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 140, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8960*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 152, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-140) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sa/csazgmq3i2hpmx7sze5sgt2eftpyywob32565yrndql2d77jvvnf.py
# Topologically Sorted Source Nodes: [input_106, input_107, input_108], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_106 => cat_13
#   input_107 => add_61, mul_91, mul_92, sub_30
#   input_108 => relu_30
# Graph fragment:
#   %cat_13 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_12, %convolution_29], 1), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_13, %unsqueeze_241), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %relu_30 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_61,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 164)
    x0 = (xindex % 64)
    x2 = xindex // 10496
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 152, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 9728*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 164, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-152) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aw/cawg3lxmco7plgpagmvj77bex3mng7r55beu76b235sgqgqdqwv6.py
# Topologically Sorted Source Nodes: [input_113, input_114, input_115], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_113 => cat_14
#   input_114 => add_65, mul_97, mul_98, sub_32
#   input_115 => relu_32
# Graph fragment:
#   %cat_14 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_13, %convolution_31], 1), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_14, %unsqueeze_257), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %unsqueeze_261), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %unsqueeze_263), kwargs = {})
#   %relu_32 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 176)
    x0 = (xindex % 64)
    x2 = xindex // 11264
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 164, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 10496*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 176, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-164) + x1) + 768*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/vz/cvzq2wuwefkqoz74ooceyyvxjdjijcehnd6eqktfjdlzs2aqnwce.py
# Topologically Sorted Source Nodes: [input_120, input_121, input_122], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_120 => cat_15
#   input_121 => add_69, mul_103, mul_104, sub_34
#   input_122 => relu_34
# Graph fragment:
#   %cat_15 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_14, %convolution_33], 1), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_15, %unsqueeze_273), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %unsqueeze_277), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_279), kwargs = {})
#   %relu_34 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_69,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 188)
    x0 = (xindex % 64)
    x2 = xindex // 12032
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 176, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 11264*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 188, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-176) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2q/c2qmvr7itigexpes5zan5kkpykx2a4ohvjpdn27gb3tsbzklqzfa.py
# Topologically Sorted Source Nodes: [input_127, input_128, input_129], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_127 => cat_16
#   input_128 => add_73, mul_109, mul_110, sub_36
#   input_129 => relu_36
# Graph fragment:
#   %cat_16 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_15, %convolution_35], 1), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_16, %unsqueeze_289), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_293), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_295), kwargs = {})
#   %relu_36 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_73,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 200)
    x0 = (xindex % 64)
    x2 = xindex // 12800
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 188, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 12032*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 200, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-188) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dz/cdzalk736dbtynhpbp2co4yykwdpoec7gqryy7pmzhn6iir637x4.py
# Topologically Sorted Source Nodes: [input_134, input_135, input_136], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_134 => cat_17
#   input_135 => add_77, mul_115, mul_116, sub_38
#   input_136 => relu_38
# Graph fragment:
#   %cat_17 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_16, %convolution_37], 1), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_17, %unsqueeze_305), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %relu_38 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 54272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 212)
    x0 = (xindex % 64)
    x2 = xindex // 13568
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 200, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 12800*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 212, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-200) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r7/cr7h6aju7zolag4boel6vvh53rp4djs4vbr5a53krpk6ytwf2xj7.py
# Topologically Sorted Source Nodes: [input_138, input_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_138 => add_79, mul_118, mul_119, sub_39
#   input_139 => relu_39
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_313), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_39 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_79,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 106)
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


# kernel path: inductor_cache/2w/c2wtrugh4u6oflh2kjjnfyz52fiwmqipsdovnubq4yafqnyo5qmf.py
# Topologically Sorted Source Nodes: [input_144, input_145, input_146], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_144 => cat_18
#   input_145 => add_83, mul_124, mul_125, sub_41
#   input_146 => relu_41
# Graph fragment:
#   %cat_18 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_38, %convolution_40], 1), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_18, %unsqueeze_329), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_333), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_83,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 118)
    x0 = (xindex % 64)
    x2 = xindex // 7552
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 106, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 6784*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 118, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-106) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i6/ci6vnhlbtl7a6gghrxb5xlc45hq7qdcetrhswh5oz7nkqpljdvhq.py
# Topologically Sorted Source Nodes: [input_151, input_152, input_153], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_151 => cat_19
#   input_152 => add_87, mul_130, mul_131, sub_43
#   input_153 => relu_43
# Graph fragment:
#   %cat_19 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_18, %convolution_42], 1), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_19, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_43 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_87,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 130)
    x0 = (xindex % 64)
    x2 = xindex // 8320
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 118, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 7552*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 130, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-118) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aa/caax6mzhdcc5kxih3mdmssz4vwvnbuzznrmxuifppchjrw3loom7.py
# Topologically Sorted Source Nodes: [input_158, input_159, input_160], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_158 => cat_20
#   input_159 => add_91, mul_136, mul_137, sub_45
#   input_160 => relu_45
# Graph fragment:
#   %cat_20 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_19, %convolution_44], 1), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_20, %unsqueeze_361), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_365), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_367), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_91,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 142)
    x0 = (xindex % 64)
    x2 = xindex // 9088
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 130, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 8320*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 142, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-130) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e7/ce75yiwh66kvejip22jcnic5a2k3mytukme4g3gz7aensk4kyn6g.py
# Topologically Sorted Source Nodes: [input_165, input_166, input_167], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_165 => cat_21
#   input_166 => add_95, mul_142, mul_143, sub_47
#   input_167 => relu_47
# Graph fragment:
#   %cat_21 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_20, %convolution_46], 1), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_21, %unsqueeze_377), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_379), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, %unsqueeze_381), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_143, %unsqueeze_383), kwargs = {})
#   %relu_47 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_95,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 154)
    x0 = (xindex % 64)
    x2 = xindex // 9856
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 142, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 9088*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 154, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-142) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g5/cg5aoctn7zp7ef5qf6bqiqewcndu4f3itoxlgtxvuupsi34fyxqp.py
# Topologically Sorted Source Nodes: [input_172, input_173, input_174], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_172 => cat_22
#   input_173 => add_99, mul_148, mul_149, sub_49
#   input_174 => relu_49
# Graph fragment:
#   %cat_22 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_21, %convolution_48], 1), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_22, %unsqueeze_393), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_397), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_399), kwargs = {})
#   %relu_49 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_99,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 166)
    x0 = (xindex % 64)
    x2 = xindex // 10624
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 154, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 9856*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 166, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-154) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gn/cgnbodtgw6mh7v44ieojuiehdnabrxvhe4goqnneux7x53jwdccw.py
# Topologically Sorted Source Nodes: [input_179, input_180, input_181], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_179 => cat_23
#   input_180 => add_103, mul_154, mul_155, sub_51
#   input_181 => relu_51
# Graph fragment:
#   %cat_23 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_22, %convolution_50], 1), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_23, %unsqueeze_409), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, %unsqueeze_413), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %unsqueeze_415), kwargs = {})
#   %relu_51 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_103,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 178)
    x0 = (xindex % 64)
    x2 = xindex // 11392
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 166, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 10624*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 178, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-166) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbrvrhlwiwwidqlsty2z2qnweyahxragcxzqjhd6ags2kygel54d.py
# Topologically Sorted Source Nodes: [input_186, input_187, input_188], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_186 => cat_24
#   input_187 => add_107, mul_160, mul_161, sub_53
#   input_188 => relu_53
# Graph fragment:
#   %cat_24 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_23, %convolution_52], 1), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_24, %unsqueeze_425), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_429), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_431), kwargs = {})
#   %relu_53 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 190)
    x0 = (xindex % 64)
    x2 = xindex // 12160
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 178, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 11392*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 190, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-178) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ho/chouyibnp6lxtrnsta3uesmlf5bxabdkd6asugtie4ljjlf5cq57.py
# Topologically Sorted Source Nodes: [input_193, input_194, input_195], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_193 => cat_25
#   input_194 => add_111, mul_166, mul_167, sub_55
#   input_195 => relu_55
# Graph fragment:
#   %cat_25 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_24, %convolution_54], 1), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_25, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
#   %relu_55 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_111,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 202)
    x0 = (xindex % 64)
    x2 = xindex // 12928
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 190, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 12160*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 202, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-190) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cm/ccmcgwvie4rbyst2nbunhxbjpp4yhmyawacqbqburttlukpb6zrd.py
# Topologically Sorted Source Nodes: [input_200, input_201, input_202], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_200 => cat_26
#   input_201 => add_115, mul_172, mul_173, sub_57
#   input_202 => relu_57
# Graph fragment:
#   %cat_26 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_25, %convolution_56], 1), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_26, %unsqueeze_457), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_461), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_463), kwargs = {})
#   %relu_57 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_115,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 54784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 214)
    x0 = (xindex % 64)
    x2 = xindex // 13696
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 12928*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 214, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-202) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hv/chvvhego5kreab62tw4bw7wq6u5fdbzrwi76xr5oozftqwtvgtwo.py
# Topologically Sorted Source Nodes: [input_207, input_208, input_209], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_207 => cat_27
#   input_208 => add_119, mul_178, mul_179, sub_59
#   input_209 => relu_59
# Graph fragment:
#   %cat_27 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_26, %convolution_58], 1), kwargs = {})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_27, %unsqueeze_473), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_475), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_477), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_479), kwargs = {})
#   %relu_59 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_119,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 226)
    x0 = (xindex % 64)
    x2 = xindex // 14464
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 214, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 13696*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 226, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-214) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nm/cnmppqhtnr6c3iwuch3y67l45ekzcyfysx7l4grc7bq72wggghc2.py
# Topologically Sorted Source Nodes: [input_214, input_215, input_216], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_214 => cat_28
#   input_215 => add_123, mul_184, mul_185, sub_61
#   input_216 => relu_61
# Graph fragment:
#   %cat_28 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_27, %convolution_60], 1), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_28, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_123,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 60928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 238)
    x0 = (xindex % 64)
    x2 = xindex // 15232
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 226, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 14464*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 238, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-226) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ch/cchmhif22l4bp3kvhrmvcsq4mcqwf5i3luvdffgta75inxt5gotd.py
# Topologically Sorted Source Nodes: [input_221, input_222, input_223], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_221 => cat_29
#   input_222 => add_127, mul_190, mul_191, sub_63
#   input_223 => relu_63
# Graph fragment:
#   %cat_29 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_28, %convolution_62], 1), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_29, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
#   %relu_63 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_127,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 250)
    x0 = (xindex % 64)
    x2 = xindex // 16000
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 238, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 15232*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 250, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-238) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5d/c5ddkcb2mvo6xlard6ujvt7jymt7driwpptop3wr4bzio3aiucqz.py
# Topologically Sorted Source Nodes: [input_228, input_229, input_230], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_228 => cat_30
#   input_229 => add_131, mul_196, mul_197, sub_65
#   input_230 => relu_65
# Graph fragment:
#   %cat_30 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_29, %convolution_64], 1), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_30, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %relu_65 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_131,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 262)
    x0 = (xindex % 64)
    x2 = xindex // 16768
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 250, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 16000*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 262, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-250) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zw/czwcnqd3bznikvqrlh7lt5ty3syiawu3cbod5vaudxsfvxsbqzrn.py
# Topologically Sorted Source Nodes: [input_235, input_236, input_237], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_235 => cat_31
#   input_236 => add_135, mul_202, mul_203, sub_67
#   input_237 => relu_67
# Graph fragment:
#   %cat_31 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_30, %convolution_66], 1), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_31, %unsqueeze_537), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_541), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_543), kwargs = {})
#   %relu_67 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 70144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 274)
    x0 = (xindex % 64)
    x2 = xindex // 17536
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 262, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 16768*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 274, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-262) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u3/cu3gvjmksjpusurgjma2gfqlmozdxedqvfmvz2ypwxxajihefkvc.py
# Topologically Sorted Source Nodes: [input_242, input_243, input_244], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_242 => cat_32
#   input_243 => add_139, mul_208, mul_209, sub_69
#   input_244 => relu_69
# Graph fragment:
#   %cat_32 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_31, %convolution_68], 1), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_32, %unsqueeze_553), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %unsqueeze_557), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_209, %unsqueeze_559), kwargs = {})
#   %relu_69 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_139,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 286)
    x0 = (xindex % 64)
    x2 = xindex // 18304
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 274, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 17536*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 286, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-274) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/co/ccoolfmnlciy3xyuuzuoqyirjtpzoxsma4w3ar746vccuqxgwskn.py
# Topologically Sorted Source Nodes: [input_249, input_250, input_251], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_249 => cat_33
#   input_250 => add_143, mul_214, mul_215, sub_71
#   input_251 => relu_71
# Graph fragment:
#   %cat_33 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_32, %convolution_70], 1), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_33, %unsqueeze_569), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, %unsqueeze_573), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %unsqueeze_575), kwargs = {})
#   %relu_71 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_143,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 298)
    x0 = (xindex % 64)
    x2 = xindex // 19072
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 286, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 18304*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 298, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-286) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/te/ctet7cwui7j4gvux55iqer7bv27f3aaacmlzvjrx3gxz3qhr7dnm.py
# Topologically Sorted Source Nodes: [input_256, input_257, input_258], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_256 => cat_34
#   input_257 => add_147, mul_220, mul_221, sub_73
#   input_258 => relu_73
# Graph fragment:
#   %cat_34 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_33, %convolution_72], 1), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_34, %unsqueeze_585), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_220, %unsqueeze_589), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_221, %unsqueeze_591), kwargs = {})
#   %relu_73 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_147,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 79360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 310)
    x0 = (xindex % 64)
    x2 = xindex // 19840
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 298, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 19072*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 310, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-298) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3k/c3kndp2ksq7ddobqfxu6u3uw7mlo3i2oahkatd3n3rswf55wbub6.py
# Topologically Sorted Source Nodes: [input_263, input_264, input_265], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_263 => cat_35
#   input_264 => add_151, mul_226, mul_227, sub_75
#   input_265 => relu_75
# Graph fragment:
#   %cat_35 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_34, %convolution_74], 1), kwargs = {})
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_35, %unsqueeze_601), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_603), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %unsqueeze_605), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %unsqueeze_607), kwargs = {})
#   %relu_75 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_151,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 82432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 322)
    x0 = (xindex % 64)
    x2 = xindex // 20608
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 310, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 19840*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 322, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-310) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ea/cea5nqn5judjhv3sgsbmlwls3ylovttyoihydkbzajkow2txoi2f.py
# Topologically Sorted Source Nodes: [input_270, input_271, input_272], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_270 => cat_36
#   input_271 => add_155, mul_232, mul_233, sub_77
#   input_272 => relu_77
# Graph fragment:
#   %cat_36 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_35, %convolution_76], 1), kwargs = {})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_36, %unsqueeze_617), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %unsqueeze_621), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %unsqueeze_623), kwargs = {})
#   %relu_77 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 85504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 334)
    x0 = (xindex % 64)
    x2 = xindex // 21376
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 322, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 20608*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 334, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-322) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/al/callwvqk7fyal5tvt7clqyscrqc6khojog3vcvfxutcomnxb7vcy.py
# Topologically Sorted Source Nodes: [input_277, input_278, input_279], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_277 => cat_37
#   input_278 => add_159, mul_238, mul_239, sub_79
#   input_279 => relu_79
# Graph fragment:
#   %cat_37 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_36, %convolution_78], 1), kwargs = {})
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_37, %unsqueeze_633), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_637), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_639), kwargs = {})
#   %relu_79 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_159,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 88576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 346)
    x0 = (xindex % 64)
    x2 = xindex // 22144
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 334, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 21376*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 346, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-334) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ts/ctshivlxo6ruh6k6cundchf5f6lwskfpl5mldkey6ptnvbw4kmk7.py
# Topologically Sorted Source Nodes: [input_284, input_285, input_286], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_284 => cat_38
#   input_285 => add_163, mul_244, mul_245, sub_81
#   input_286 => relu_81
# Graph fragment:
#   %cat_38 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_37, %convolution_80], 1), kwargs = {})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_38, %unsqueeze_649), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %unsqueeze_653), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %unsqueeze_655), kwargs = {})
#   %relu_81 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_163,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 91648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 358)
    x0 = (xindex % 64)
    x2 = xindex // 22912
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 346, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 22144*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 358, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-346) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y4/cy4b5tib3hocv3ipnlv5rban37pi6aq6izwol7cfdxwmtbjefoff.py
# Topologically Sorted Source Nodes: [input_291, input_292, input_293], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_291 => cat_39
#   input_292 => add_167, mul_250, mul_251, sub_83
#   input_293 => relu_83
# Graph fragment:
#   %cat_39 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_38, %convolution_82], 1), kwargs = {})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_39, %unsqueeze_665), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_250, %unsqueeze_669), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_251, %unsqueeze_671), kwargs = {})
#   %relu_83 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_167,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 370)
    x0 = (xindex % 64)
    x2 = xindex // 23680
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 358, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 22912*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 370, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-358) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fb/cfbcd7t32wvpoubry74hocgpuymu4dlqfv4wgxoes23vljxfavsj.py
# Topologically Sorted Source Nodes: [input_298, input_299, input_300], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_298 => cat_40
#   input_299 => add_171, mul_256, mul_257, sub_85
#   input_300 => relu_85
# Graph fragment:
#   %cat_40 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_39, %convolution_84], 1), kwargs = {})
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_40, %unsqueeze_681), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_685), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_687), kwargs = {})
#   %relu_85 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_171,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 97792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 382)
    x0 = (xindex % 64)
    x2 = xindex // 24448
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 370, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 23680*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 382, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-370) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cik3i7uh4esoo5yx52imdjwjjronzvop57qpyk5lypxfayxkrtcv.py
# Topologically Sorted Source Nodes: [input_305, input_306, input_307], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_305 => cat_41
#   input_306 => add_175, mul_262, mul_263, sub_87
#   input_307 => relu_87
# Graph fragment:
#   %cat_41 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_40, %convolution_86], 1), kwargs = {})
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_41, %unsqueeze_697), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_701), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_703), kwargs = {})
#   %relu_87 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_175,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 394)
    x0 = (xindex % 64)
    x2 = xindex // 25216
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 382, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 24448*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 394, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-382) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h5/ch5raqrcw7sercedfl4mnlxhrdcn6r5qtwvylpsf7tvtfpwjxfrd.py
# Topologically Sorted Source Nodes: [input_309, input_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_309 => add_177, mul_265, mul_266, sub_88
#   input_310 => relu_88
# Graph fragment:
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_705), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_709), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_711), kwargs = {})
#   %relu_88 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_177,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 64) % 197)
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


# kernel path: inductor_cache/le/cleagco4ut3lhfmy5toxfc3ug2apbagi6nmwdycbjedmk7yb57bj.py
# Topologically Sorted Source Nodes: [input_315, input_316, input_317], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_315 => cat_42
#   input_316 => add_181, mul_271, mul_272, sub_90
#   input_317 => relu_90
# Graph fragment:
#   %cat_42 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_87, %convolution_89], 1), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_42, %unsqueeze_721), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_725), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_727), kwargs = {})
#   %relu_90 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_181,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 209)
    x0 = (xindex % 64)
    x2 = xindex // 13376
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 197, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 12608*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 209, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-197) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ih/cihioffjrqz7txxz3leb7loi6o2lp5dtw3rdvazfvxvue57gcy2s.py
# Topologically Sorted Source Nodes: [input_322, input_323, input_324], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_322 => cat_43
#   input_323 => add_185, mul_277, mul_278, sub_92
#   input_324 => relu_92
# Graph fragment:
#   %cat_43 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_42, %convolution_91], 1), kwargs = {})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_43, %unsqueeze_737), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_277, %unsqueeze_741), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_278, %unsqueeze_743), kwargs = {})
#   %relu_92 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_185,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_51', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 56576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 221)
    x0 = (xindex % 64)
    x2 = xindex // 14144
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 209, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 13376*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 221, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-209) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yj/cyjr6imdfxch3fanf7xhgizlqovo2ztycntx4g3g2hwbquia4rfp.py
# Topologically Sorted Source Nodes: [input_329, input_330, input_331], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_329 => cat_44
#   input_330 => add_189, mul_283, mul_284, sub_94
#   input_331 => relu_94
# Graph fragment:
#   %cat_44 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_43, %convolution_93], 1), kwargs = {})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_44, %unsqueeze_753), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_757), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_759), kwargs = {})
#   %relu_94 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_189,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 59648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 233)
    x0 = (xindex % 64)
    x2 = xindex // 14912
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 221, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 14144*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 233, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-221) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e6/ce6sso47ljrg6fcruohshuiuj3gq4k7utpmccoahc5a3oyrg3bmc.py
# Topologically Sorted Source Nodes: [input_336, input_337, input_338], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_336 => cat_45
#   input_337 => add_193, mul_289, mul_290, sub_96
#   input_338 => relu_96
# Graph fragment:
#   %cat_45 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_44, %convolution_95], 1), kwargs = {})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_45, %unsqueeze_769), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_773), kwargs = {})
#   %add_193 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_775), kwargs = {})
#   %relu_96 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_193,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 245)
    x0 = (xindex % 64)
    x2 = xindex // 15680
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 233, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 14912*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 245, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-233) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ml/cml532ypgvi46nua3wr2q5f2byl3chmy2feoszilnph7mvbrnazl.py
# Topologically Sorted Source Nodes: [input_343, input_344, input_345], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_343 => cat_46
#   input_344 => add_197, mul_295, mul_296, sub_98
#   input_345 => relu_98
# Graph fragment:
#   %cat_46 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_45, %convolution_97], 1), kwargs = {})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_46, %unsqueeze_785), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %unsqueeze_787), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_295, %unsqueeze_789), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_296, %unsqueeze_791), kwargs = {})
#   %relu_98 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_197,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 257)
    x0 = (xindex % 64)
    x2 = xindex // 16448
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 245, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 15680*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 257, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-245) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rn/crnshwgg2vqsgzxkjeyromi2njd5oqj4s3q5rhynga7smoukowqe.py
# Topologically Sorted Source Nodes: [input_350, input_351, input_352], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_350 => cat_47
#   input_351 => add_201, mul_301, mul_302, sub_100
#   input_352 => relu_100
# Graph fragment:
#   %cat_47 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_46, %convolution_99], 1), kwargs = {})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_47, %unsqueeze_801), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_805), kwargs = {})
#   %add_201 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_807), kwargs = {})
#   %relu_100 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_201,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 269)
    x0 = (xindex % 64)
    x2 = xindex // 17216
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 257, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 16448*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 269, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-257) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rr/crrsxq7qh2b4lpl3me764hvk5uiswczuoczeu5ttbm45jkw36m4a.py
# Topologically Sorted Source Nodes: [input_357, input_358, input_359], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_357 => cat_48
#   input_358 => add_205, mul_307, mul_308, sub_102
#   input_359 => relu_102
# Graph fragment:
#   %cat_48 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_47, %convolution_101], 1), kwargs = {})
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_48, %unsqueeze_817), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %unsqueeze_821), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %unsqueeze_823), kwargs = {})
#   %relu_102 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_205,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 71936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 281)
    x0 = (xindex % 64)
    x2 = xindex // 17984
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 269, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 17216*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 281, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-269) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/me/cmekypmvxxfpe7lwda7bh753qu4e6uiekbcglfm4yljhsrki6yjm.py
# Topologically Sorted Source Nodes: [input_364, input_365, input_366], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_364 => cat_49
#   input_365 => add_209, mul_313, mul_314, sub_104
#   input_366 => relu_104
# Graph fragment:
#   %cat_49 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_48, %convolution_103], 1), kwargs = {})
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_49, %unsqueeze_833), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_313, %unsqueeze_837), kwargs = {})
#   %add_209 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %unsqueeze_839), kwargs = {})
#   %relu_104 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_209,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 293)
    x0 = (xindex % 64)
    x2 = xindex // 18752
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 281, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 17984*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 293, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-281) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5j/c5jlduxyvn5gdkbvt3kgjo2cyngbtelmw7wtqd5aq6bifhytegwl.py
# Topologically Sorted Source Nodes: [input_371, input_372, input_373], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_371 => cat_50
#   input_372 => add_213, mul_319, mul_320, sub_106
#   input_373 => relu_106
# Graph fragment:
#   %cat_50 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_49, %convolution_105], 1), kwargs = {})
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_50, %unsqueeze_849), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %unsqueeze_851), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_319, %unsqueeze_853), kwargs = {})
#   %add_213 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_320, %unsqueeze_855), kwargs = {})
#   %relu_106 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_213,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 305)
    x0 = (xindex % 64)
    x2 = xindex // 19520
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 293, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 18752*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 305, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-293) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2y/c2ylnyrm34l47ufm427ogs6mqed3gbcmksxwalvytdb4kfmuusne.py
# Topologically Sorted Source Nodes: [input_378, input_379, input_380], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_378 => cat_51
#   input_379 => add_217, mul_325, mul_326, sub_108
#   input_380 => relu_108
# Graph fragment:
#   %cat_51 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_50, %convolution_107], 1), kwargs = {})
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_51, %unsqueeze_865), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_867), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_325, %unsqueeze_869), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %unsqueeze_871), kwargs = {})
#   %relu_108 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_217,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 317)
    x0 = (xindex % 64)
    x2 = xindex // 20288
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 305, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 19520*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 317, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-305) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ds/cdss6flu6zgbp2wf5xdtdcr7vncotdg7ia6gwjhmiaulnzu62fcw.py
# Topologically Sorted Source Nodes: [input_385, input_386, input_387], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_385 => cat_52
#   input_386 => add_221, mul_331, mul_332, sub_110
#   input_387 => relu_110
# Graph fragment:
#   %cat_52 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_51, %convolution_109], 1), kwargs = {})
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_52, %unsqueeze_881), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %unsqueeze_883), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %unsqueeze_885), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_332, %unsqueeze_887), kwargs = {})
#   %relu_110 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_221,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 84224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 329)
    x0 = (xindex % 64)
    x2 = xindex // 21056
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 317, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 20288*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 329, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-317) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ls/clsu7jrd5ius4kn6ek7sl3hfhobfxfh2opb3usidfjwavg3rimlw.py
# Topologically Sorted Source Nodes: [input_392, input_393, input_394], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_392 => cat_53
#   input_393 => add_225, mul_337, mul_338, sub_112
#   input_394 => relu_112
# Graph fragment:
#   %cat_53 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_52, %convolution_111], 1), kwargs = {})
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_53, %unsqueeze_897), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %unsqueeze_899), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %unsqueeze_901), kwargs = {})
#   %add_225 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_338, %unsqueeze_903), kwargs = {})
#   %relu_112 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_225,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 341)
    x0 = (xindex % 64)
    x2 = xindex // 21824
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 329, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 21056*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 341, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-329) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pz/cpzfs2q6qwhdopmpiwhw2kztsdlrb5vcfa2iesj5tq5ru7pzleby.py
# Topologically Sorted Source Nodes: [input_399, input_400, input_401], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_399 => cat_54
#   input_400 => add_229, mul_343, mul_344, sub_114
#   input_401 => relu_114
# Graph fragment:
#   %cat_54 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_53, %convolution_113], 1), kwargs = {})
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_54, %unsqueeze_913), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_917), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_919), kwargs = {})
#   %relu_114 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_229,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 353)
    x0 = (xindex % 64)
    x2 = xindex // 22592
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 341, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 21824*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 353, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-341) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t5/ct5zp5ka7qnv3iv7hfh4ljef42ztzn4oyr64llvsyoxbegklve7y.py
# Topologically Sorted Source Nodes: [input_406, input_407, input_408], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_406 => cat_55
#   input_407 => add_233, mul_349, mul_350, sub_116
#   input_408 => relu_116
# Graph fragment:
#   %cat_55 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_54, %convolution_115], 1), kwargs = {})
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_55, %unsqueeze_929), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %unsqueeze_931), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_933), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_935), kwargs = {})
#   %relu_116 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_233,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 93440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 365)
    x0 = (xindex % 64)
    x2 = xindex // 23360
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 353, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 22592*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 365, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-353) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rw/crw2jkmaek3tb4pegvwmprkv4nqv5sjrkni3fjdsqhhs27xevzip.py
# Topologically Sorted Source Nodes: [input_413, input_414, input_415], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_413 => cat_56
#   input_414 => add_237, mul_355, mul_356, sub_118
#   input_415 => relu_118
# Graph fragment:
#   %cat_56 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_55, %convolution_117], 1), kwargs = {})
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_56, %unsqueeze_945), kwargs = {})
#   %mul_355 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %unsqueeze_947), kwargs = {})
#   %mul_356 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_355, %unsqueeze_949), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_356, %unsqueeze_951), kwargs = {})
#   %relu_118 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_237,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % 377)
    x0 = (xindex % 64)
    x2 = xindex // 24128
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 365, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 23360*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 377, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*((-365) + x1) + 768*x2), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w3/cw3gxgbzjbku56horc7j7citrdh5al4hjp7vjzktnr7qdycf5y6w.py
# Topologically Sorted Source Nodes: [input_420, input_421, out, adaptive_avg_pool2d], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => mean
#   input_420 => cat_57
#   input_421 => add_241, mul_361, mul_362, sub_120
#   out => relu_120
# Graph fragment:
#   %cat_57 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_56, %convolution_119], 1), kwargs = {})
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_57, %unsqueeze_961), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_963), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_361, %unsqueeze_965), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_362, %unsqueeze_967), kwargs = {})
#   %relu_120 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_241,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_120, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_65 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_65', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1556
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = (xindex % 389)
    r2 = rindex
    x1 = xindex // 389
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 377, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + 64*(x0) + 24128*x1), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 389, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + 64*((-377) + x0) + 768*x1), tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1, 1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1, 1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 64.0
    tmp33 = tmp31 / tmp32
    tl.store(out_ptr0 + (r2 + 64*x3), tmp10, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (48, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_12, (48, ), (1, ))
    assert_size_stride(primals_13, (48, ), (1, ))
    assert_size_stride(primals_14, (48, ), (1, ))
    assert_size_stride(primals_15, (48, ), (1, ))
    assert_size_stride(primals_16, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_17, (76, ), (1, ))
    assert_size_stride(primals_18, (76, ), (1, ))
    assert_size_stride(primals_19, (76, ), (1, ))
    assert_size_stride(primals_20, (76, ), (1, ))
    assert_size_stride(primals_21, (48, 76, 1, 1), (76, 1, 1, 1))
    assert_size_stride(primals_22, (48, ), (1, ))
    assert_size_stride(primals_23, (48, ), (1, ))
    assert_size_stride(primals_24, (48, ), (1, ))
    assert_size_stride(primals_25, (48, ), (1, ))
    assert_size_stride(primals_26, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_27, (88, ), (1, ))
    assert_size_stride(primals_28, (88, ), (1, ))
    assert_size_stride(primals_29, (88, ), (1, ))
    assert_size_stride(primals_30, (88, ), (1, ))
    assert_size_stride(primals_31, (48, 88, 1, 1), (88, 1, 1, 1))
    assert_size_stride(primals_32, (48, ), (1, ))
    assert_size_stride(primals_33, (48, ), (1, ))
    assert_size_stride(primals_34, (48, ), (1, ))
    assert_size_stride(primals_35, (48, ), (1, ))
    assert_size_stride(primals_36, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_37, (100, ), (1, ))
    assert_size_stride(primals_38, (100, ), (1, ))
    assert_size_stride(primals_39, (100, ), (1, ))
    assert_size_stride(primals_40, (100, ), (1, ))
    assert_size_stride(primals_41, (48, 100, 1, 1), (100, 1, 1, 1))
    assert_size_stride(primals_42, (48, ), (1, ))
    assert_size_stride(primals_43, (48, ), (1, ))
    assert_size_stride(primals_44, (48, ), (1, ))
    assert_size_stride(primals_45, (48, ), (1, ))
    assert_size_stride(primals_46, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_47, (112, ), (1, ))
    assert_size_stride(primals_48, (112, ), (1, ))
    assert_size_stride(primals_49, (112, ), (1, ))
    assert_size_stride(primals_50, (112, ), (1, ))
    assert_size_stride(primals_51, (48, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_52, (48, ), (1, ))
    assert_size_stride(primals_53, (48, ), (1, ))
    assert_size_stride(primals_54, (48, ), (1, ))
    assert_size_stride(primals_55, (48, ), (1, ))
    assert_size_stride(primals_56, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_57, (124, ), (1, ))
    assert_size_stride(primals_58, (124, ), (1, ))
    assert_size_stride(primals_59, (124, ), (1, ))
    assert_size_stride(primals_60, (124, ), (1, ))
    assert_size_stride(primals_61, (48, 124, 1, 1), (124, 1, 1, 1))
    assert_size_stride(primals_62, (48, ), (1, ))
    assert_size_stride(primals_63, (48, ), (1, ))
    assert_size_stride(primals_64, (48, ), (1, ))
    assert_size_stride(primals_65, (48, ), (1, ))
    assert_size_stride(primals_66, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_67, (136, ), (1, ))
    assert_size_stride(primals_68, (136, ), (1, ))
    assert_size_stride(primals_69, (136, ), (1, ))
    assert_size_stride(primals_70, (136, ), (1, ))
    assert_size_stride(primals_71, (68, 136, 1, 1), (136, 1, 1, 1))
    assert_size_stride(primals_72, (68, ), (1, ))
    assert_size_stride(primals_73, (68, ), (1, ))
    assert_size_stride(primals_74, (68, ), (1, ))
    assert_size_stride(primals_75, (68, ), (1, ))
    assert_size_stride(primals_76, (48, 68, 1, 1), (68, 1, 1, 1))
    assert_size_stride(primals_77, (48, ), (1, ))
    assert_size_stride(primals_78, (48, ), (1, ))
    assert_size_stride(primals_79, (48, ), (1, ))
    assert_size_stride(primals_80, (48, ), (1, ))
    assert_size_stride(primals_81, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_82, (80, ), (1, ))
    assert_size_stride(primals_83, (80, ), (1, ))
    assert_size_stride(primals_84, (80, ), (1, ))
    assert_size_stride(primals_85, (80, ), (1, ))
    assert_size_stride(primals_86, (48, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_87, (48, ), (1, ))
    assert_size_stride(primals_88, (48, ), (1, ))
    assert_size_stride(primals_89, (48, ), (1, ))
    assert_size_stride(primals_90, (48, ), (1, ))
    assert_size_stride(primals_91, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_92, (92, ), (1, ))
    assert_size_stride(primals_93, (92, ), (1, ))
    assert_size_stride(primals_94, (92, ), (1, ))
    assert_size_stride(primals_95, (92, ), (1, ))
    assert_size_stride(primals_96, (48, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_97, (48, ), (1, ))
    assert_size_stride(primals_98, (48, ), (1, ))
    assert_size_stride(primals_99, (48, ), (1, ))
    assert_size_stride(primals_100, (48, ), (1, ))
    assert_size_stride(primals_101, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_102, (104, ), (1, ))
    assert_size_stride(primals_103, (104, ), (1, ))
    assert_size_stride(primals_104, (104, ), (1, ))
    assert_size_stride(primals_105, (104, ), (1, ))
    assert_size_stride(primals_106, (48, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_107, (48, ), (1, ))
    assert_size_stride(primals_108, (48, ), (1, ))
    assert_size_stride(primals_109, (48, ), (1, ))
    assert_size_stride(primals_110, (48, ), (1, ))
    assert_size_stride(primals_111, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_112, (116, ), (1, ))
    assert_size_stride(primals_113, (116, ), (1, ))
    assert_size_stride(primals_114, (116, ), (1, ))
    assert_size_stride(primals_115, (116, ), (1, ))
    assert_size_stride(primals_116, (48, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_117, (48, ), (1, ))
    assert_size_stride(primals_118, (48, ), (1, ))
    assert_size_stride(primals_119, (48, ), (1, ))
    assert_size_stride(primals_120, (48, ), (1, ))
    assert_size_stride(primals_121, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (48, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_127, (48, ), (1, ))
    assert_size_stride(primals_128, (48, ), (1, ))
    assert_size_stride(primals_129, (48, ), (1, ))
    assert_size_stride(primals_130, (48, ), (1, ))
    assert_size_stride(primals_131, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_132, (140, ), (1, ))
    assert_size_stride(primals_133, (140, ), (1, ))
    assert_size_stride(primals_134, (140, ), (1, ))
    assert_size_stride(primals_135, (140, ), (1, ))
    assert_size_stride(primals_136, (48, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(primals_137, (48, ), (1, ))
    assert_size_stride(primals_138, (48, ), (1, ))
    assert_size_stride(primals_139, (48, ), (1, ))
    assert_size_stride(primals_140, (48, ), (1, ))
    assert_size_stride(primals_141, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_142, (152, ), (1, ))
    assert_size_stride(primals_143, (152, ), (1, ))
    assert_size_stride(primals_144, (152, ), (1, ))
    assert_size_stride(primals_145, (152, ), (1, ))
    assert_size_stride(primals_146, (48, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_147, (48, ), (1, ))
    assert_size_stride(primals_148, (48, ), (1, ))
    assert_size_stride(primals_149, (48, ), (1, ))
    assert_size_stride(primals_150, (48, ), (1, ))
    assert_size_stride(primals_151, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_152, (164, ), (1, ))
    assert_size_stride(primals_153, (164, ), (1, ))
    assert_size_stride(primals_154, (164, ), (1, ))
    assert_size_stride(primals_155, (164, ), (1, ))
    assert_size_stride(primals_156, (48, 164, 1, 1), (164, 1, 1, 1))
    assert_size_stride(primals_157, (48, ), (1, ))
    assert_size_stride(primals_158, (48, ), (1, ))
    assert_size_stride(primals_159, (48, ), (1, ))
    assert_size_stride(primals_160, (48, ), (1, ))
    assert_size_stride(primals_161, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_162, (176, ), (1, ))
    assert_size_stride(primals_163, (176, ), (1, ))
    assert_size_stride(primals_164, (176, ), (1, ))
    assert_size_stride(primals_165, (176, ), (1, ))
    assert_size_stride(primals_166, (48, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_167, (48, ), (1, ))
    assert_size_stride(primals_168, (48, ), (1, ))
    assert_size_stride(primals_169, (48, ), (1, ))
    assert_size_stride(primals_170, (48, ), (1, ))
    assert_size_stride(primals_171, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_172, (188, ), (1, ))
    assert_size_stride(primals_173, (188, ), (1, ))
    assert_size_stride(primals_174, (188, ), (1, ))
    assert_size_stride(primals_175, (188, ), (1, ))
    assert_size_stride(primals_176, (48, 188, 1, 1), (188, 1, 1, 1))
    assert_size_stride(primals_177, (48, ), (1, ))
    assert_size_stride(primals_178, (48, ), (1, ))
    assert_size_stride(primals_179, (48, ), (1, ))
    assert_size_stride(primals_180, (48, ), (1, ))
    assert_size_stride(primals_181, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_182, (200, ), (1, ))
    assert_size_stride(primals_183, (200, ), (1, ))
    assert_size_stride(primals_184, (200, ), (1, ))
    assert_size_stride(primals_185, (200, ), (1, ))
    assert_size_stride(primals_186, (48, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_187, (48, ), (1, ))
    assert_size_stride(primals_188, (48, ), (1, ))
    assert_size_stride(primals_189, (48, ), (1, ))
    assert_size_stride(primals_190, (48, ), (1, ))
    assert_size_stride(primals_191, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_192, (212, ), (1, ))
    assert_size_stride(primals_193, (212, ), (1, ))
    assert_size_stride(primals_194, (212, ), (1, ))
    assert_size_stride(primals_195, (212, ), (1, ))
    assert_size_stride(primals_196, (106, 212, 1, 1), (212, 1, 1, 1))
    assert_size_stride(primals_197, (106, ), (1, ))
    assert_size_stride(primals_198, (106, ), (1, ))
    assert_size_stride(primals_199, (106, ), (1, ))
    assert_size_stride(primals_200, (106, ), (1, ))
    assert_size_stride(primals_201, (48, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(primals_202, (48, ), (1, ))
    assert_size_stride(primals_203, (48, ), (1, ))
    assert_size_stride(primals_204, (48, ), (1, ))
    assert_size_stride(primals_205, (48, ), (1, ))
    assert_size_stride(primals_206, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_207, (118, ), (1, ))
    assert_size_stride(primals_208, (118, ), (1, ))
    assert_size_stride(primals_209, (118, ), (1, ))
    assert_size_stride(primals_210, (118, ), (1, ))
    assert_size_stride(primals_211, (48, 118, 1, 1), (118, 1, 1, 1))
    assert_size_stride(primals_212, (48, ), (1, ))
    assert_size_stride(primals_213, (48, ), (1, ))
    assert_size_stride(primals_214, (48, ), (1, ))
    assert_size_stride(primals_215, (48, ), (1, ))
    assert_size_stride(primals_216, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_217, (130, ), (1, ))
    assert_size_stride(primals_218, (130, ), (1, ))
    assert_size_stride(primals_219, (130, ), (1, ))
    assert_size_stride(primals_220, (130, ), (1, ))
    assert_size_stride(primals_221, (48, 130, 1, 1), (130, 1, 1, 1))
    assert_size_stride(primals_222, (48, ), (1, ))
    assert_size_stride(primals_223, (48, ), (1, ))
    assert_size_stride(primals_224, (48, ), (1, ))
    assert_size_stride(primals_225, (48, ), (1, ))
    assert_size_stride(primals_226, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_227, (142, ), (1, ))
    assert_size_stride(primals_228, (142, ), (1, ))
    assert_size_stride(primals_229, (142, ), (1, ))
    assert_size_stride(primals_230, (142, ), (1, ))
    assert_size_stride(primals_231, (48, 142, 1, 1), (142, 1, 1, 1))
    assert_size_stride(primals_232, (48, ), (1, ))
    assert_size_stride(primals_233, (48, ), (1, ))
    assert_size_stride(primals_234, (48, ), (1, ))
    assert_size_stride(primals_235, (48, ), (1, ))
    assert_size_stride(primals_236, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_237, (154, ), (1, ))
    assert_size_stride(primals_238, (154, ), (1, ))
    assert_size_stride(primals_239, (154, ), (1, ))
    assert_size_stride(primals_240, (154, ), (1, ))
    assert_size_stride(primals_241, (48, 154, 1, 1), (154, 1, 1, 1))
    assert_size_stride(primals_242, (48, ), (1, ))
    assert_size_stride(primals_243, (48, ), (1, ))
    assert_size_stride(primals_244, (48, ), (1, ))
    assert_size_stride(primals_245, (48, ), (1, ))
    assert_size_stride(primals_246, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_247, (166, ), (1, ))
    assert_size_stride(primals_248, (166, ), (1, ))
    assert_size_stride(primals_249, (166, ), (1, ))
    assert_size_stride(primals_250, (166, ), (1, ))
    assert_size_stride(primals_251, (48, 166, 1, 1), (166, 1, 1, 1))
    assert_size_stride(primals_252, (48, ), (1, ))
    assert_size_stride(primals_253, (48, ), (1, ))
    assert_size_stride(primals_254, (48, ), (1, ))
    assert_size_stride(primals_255, (48, ), (1, ))
    assert_size_stride(primals_256, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_257, (178, ), (1, ))
    assert_size_stride(primals_258, (178, ), (1, ))
    assert_size_stride(primals_259, (178, ), (1, ))
    assert_size_stride(primals_260, (178, ), (1, ))
    assert_size_stride(primals_261, (48, 178, 1, 1), (178, 1, 1, 1))
    assert_size_stride(primals_262, (48, ), (1, ))
    assert_size_stride(primals_263, (48, ), (1, ))
    assert_size_stride(primals_264, (48, ), (1, ))
    assert_size_stride(primals_265, (48, ), (1, ))
    assert_size_stride(primals_266, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_267, (190, ), (1, ))
    assert_size_stride(primals_268, (190, ), (1, ))
    assert_size_stride(primals_269, (190, ), (1, ))
    assert_size_stride(primals_270, (190, ), (1, ))
    assert_size_stride(primals_271, (48, 190, 1, 1), (190, 1, 1, 1))
    assert_size_stride(primals_272, (48, ), (1, ))
    assert_size_stride(primals_273, (48, ), (1, ))
    assert_size_stride(primals_274, (48, ), (1, ))
    assert_size_stride(primals_275, (48, ), (1, ))
    assert_size_stride(primals_276, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_277, (202, ), (1, ))
    assert_size_stride(primals_278, (202, ), (1, ))
    assert_size_stride(primals_279, (202, ), (1, ))
    assert_size_stride(primals_280, (202, ), (1, ))
    assert_size_stride(primals_281, (48, 202, 1, 1), (202, 1, 1, 1))
    assert_size_stride(primals_282, (48, ), (1, ))
    assert_size_stride(primals_283, (48, ), (1, ))
    assert_size_stride(primals_284, (48, ), (1, ))
    assert_size_stride(primals_285, (48, ), (1, ))
    assert_size_stride(primals_286, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_287, (214, ), (1, ))
    assert_size_stride(primals_288, (214, ), (1, ))
    assert_size_stride(primals_289, (214, ), (1, ))
    assert_size_stride(primals_290, (214, ), (1, ))
    assert_size_stride(primals_291, (48, 214, 1, 1), (214, 1, 1, 1))
    assert_size_stride(primals_292, (48, ), (1, ))
    assert_size_stride(primals_293, (48, ), (1, ))
    assert_size_stride(primals_294, (48, ), (1, ))
    assert_size_stride(primals_295, (48, ), (1, ))
    assert_size_stride(primals_296, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_297, (226, ), (1, ))
    assert_size_stride(primals_298, (226, ), (1, ))
    assert_size_stride(primals_299, (226, ), (1, ))
    assert_size_stride(primals_300, (226, ), (1, ))
    assert_size_stride(primals_301, (48, 226, 1, 1), (226, 1, 1, 1))
    assert_size_stride(primals_302, (48, ), (1, ))
    assert_size_stride(primals_303, (48, ), (1, ))
    assert_size_stride(primals_304, (48, ), (1, ))
    assert_size_stride(primals_305, (48, ), (1, ))
    assert_size_stride(primals_306, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_307, (238, ), (1, ))
    assert_size_stride(primals_308, (238, ), (1, ))
    assert_size_stride(primals_309, (238, ), (1, ))
    assert_size_stride(primals_310, (238, ), (1, ))
    assert_size_stride(primals_311, (48, 238, 1, 1), (238, 1, 1, 1))
    assert_size_stride(primals_312, (48, ), (1, ))
    assert_size_stride(primals_313, (48, ), (1, ))
    assert_size_stride(primals_314, (48, ), (1, ))
    assert_size_stride(primals_315, (48, ), (1, ))
    assert_size_stride(primals_316, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_317, (250, ), (1, ))
    assert_size_stride(primals_318, (250, ), (1, ))
    assert_size_stride(primals_319, (250, ), (1, ))
    assert_size_stride(primals_320, (250, ), (1, ))
    assert_size_stride(primals_321, (48, 250, 1, 1), (250, 1, 1, 1))
    assert_size_stride(primals_322, (48, ), (1, ))
    assert_size_stride(primals_323, (48, ), (1, ))
    assert_size_stride(primals_324, (48, ), (1, ))
    assert_size_stride(primals_325, (48, ), (1, ))
    assert_size_stride(primals_326, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_327, (262, ), (1, ))
    assert_size_stride(primals_328, (262, ), (1, ))
    assert_size_stride(primals_329, (262, ), (1, ))
    assert_size_stride(primals_330, (262, ), (1, ))
    assert_size_stride(primals_331, (48, 262, 1, 1), (262, 1, 1, 1))
    assert_size_stride(primals_332, (48, ), (1, ))
    assert_size_stride(primals_333, (48, ), (1, ))
    assert_size_stride(primals_334, (48, ), (1, ))
    assert_size_stride(primals_335, (48, ), (1, ))
    assert_size_stride(primals_336, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_337, (274, ), (1, ))
    assert_size_stride(primals_338, (274, ), (1, ))
    assert_size_stride(primals_339, (274, ), (1, ))
    assert_size_stride(primals_340, (274, ), (1, ))
    assert_size_stride(primals_341, (48, 274, 1, 1), (274, 1, 1, 1))
    assert_size_stride(primals_342, (48, ), (1, ))
    assert_size_stride(primals_343, (48, ), (1, ))
    assert_size_stride(primals_344, (48, ), (1, ))
    assert_size_stride(primals_345, (48, ), (1, ))
    assert_size_stride(primals_346, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_347, (286, ), (1, ))
    assert_size_stride(primals_348, (286, ), (1, ))
    assert_size_stride(primals_349, (286, ), (1, ))
    assert_size_stride(primals_350, (286, ), (1, ))
    assert_size_stride(primals_351, (48, 286, 1, 1), (286, 1, 1, 1))
    assert_size_stride(primals_352, (48, ), (1, ))
    assert_size_stride(primals_353, (48, ), (1, ))
    assert_size_stride(primals_354, (48, ), (1, ))
    assert_size_stride(primals_355, (48, ), (1, ))
    assert_size_stride(primals_356, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_357, (298, ), (1, ))
    assert_size_stride(primals_358, (298, ), (1, ))
    assert_size_stride(primals_359, (298, ), (1, ))
    assert_size_stride(primals_360, (298, ), (1, ))
    assert_size_stride(primals_361, (48, 298, 1, 1), (298, 1, 1, 1))
    assert_size_stride(primals_362, (48, ), (1, ))
    assert_size_stride(primals_363, (48, ), (1, ))
    assert_size_stride(primals_364, (48, ), (1, ))
    assert_size_stride(primals_365, (48, ), (1, ))
    assert_size_stride(primals_366, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_367, (310, ), (1, ))
    assert_size_stride(primals_368, (310, ), (1, ))
    assert_size_stride(primals_369, (310, ), (1, ))
    assert_size_stride(primals_370, (310, ), (1, ))
    assert_size_stride(primals_371, (48, 310, 1, 1), (310, 1, 1, 1))
    assert_size_stride(primals_372, (48, ), (1, ))
    assert_size_stride(primals_373, (48, ), (1, ))
    assert_size_stride(primals_374, (48, ), (1, ))
    assert_size_stride(primals_375, (48, ), (1, ))
    assert_size_stride(primals_376, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_377, (322, ), (1, ))
    assert_size_stride(primals_378, (322, ), (1, ))
    assert_size_stride(primals_379, (322, ), (1, ))
    assert_size_stride(primals_380, (322, ), (1, ))
    assert_size_stride(primals_381, (48, 322, 1, 1), (322, 1, 1, 1))
    assert_size_stride(primals_382, (48, ), (1, ))
    assert_size_stride(primals_383, (48, ), (1, ))
    assert_size_stride(primals_384, (48, ), (1, ))
    assert_size_stride(primals_385, (48, ), (1, ))
    assert_size_stride(primals_386, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_387, (334, ), (1, ))
    assert_size_stride(primals_388, (334, ), (1, ))
    assert_size_stride(primals_389, (334, ), (1, ))
    assert_size_stride(primals_390, (334, ), (1, ))
    assert_size_stride(primals_391, (48, 334, 1, 1), (334, 1, 1, 1))
    assert_size_stride(primals_392, (48, ), (1, ))
    assert_size_stride(primals_393, (48, ), (1, ))
    assert_size_stride(primals_394, (48, ), (1, ))
    assert_size_stride(primals_395, (48, ), (1, ))
    assert_size_stride(primals_396, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_397, (346, ), (1, ))
    assert_size_stride(primals_398, (346, ), (1, ))
    assert_size_stride(primals_399, (346, ), (1, ))
    assert_size_stride(primals_400, (346, ), (1, ))
    assert_size_stride(primals_401, (48, 346, 1, 1), (346, 1, 1, 1))
    assert_size_stride(primals_402, (48, ), (1, ))
    assert_size_stride(primals_403, (48, ), (1, ))
    assert_size_stride(primals_404, (48, ), (1, ))
    assert_size_stride(primals_405, (48, ), (1, ))
    assert_size_stride(primals_406, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_407, (358, ), (1, ))
    assert_size_stride(primals_408, (358, ), (1, ))
    assert_size_stride(primals_409, (358, ), (1, ))
    assert_size_stride(primals_410, (358, ), (1, ))
    assert_size_stride(primals_411, (48, 358, 1, 1), (358, 1, 1, 1))
    assert_size_stride(primals_412, (48, ), (1, ))
    assert_size_stride(primals_413, (48, ), (1, ))
    assert_size_stride(primals_414, (48, ), (1, ))
    assert_size_stride(primals_415, (48, ), (1, ))
    assert_size_stride(primals_416, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_417, (370, ), (1, ))
    assert_size_stride(primals_418, (370, ), (1, ))
    assert_size_stride(primals_419, (370, ), (1, ))
    assert_size_stride(primals_420, (370, ), (1, ))
    assert_size_stride(primals_421, (48, 370, 1, 1), (370, 1, 1, 1))
    assert_size_stride(primals_422, (48, ), (1, ))
    assert_size_stride(primals_423, (48, ), (1, ))
    assert_size_stride(primals_424, (48, ), (1, ))
    assert_size_stride(primals_425, (48, ), (1, ))
    assert_size_stride(primals_426, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_427, (382, ), (1, ))
    assert_size_stride(primals_428, (382, ), (1, ))
    assert_size_stride(primals_429, (382, ), (1, ))
    assert_size_stride(primals_430, (382, ), (1, ))
    assert_size_stride(primals_431, (48, 382, 1, 1), (382, 1, 1, 1))
    assert_size_stride(primals_432, (48, ), (1, ))
    assert_size_stride(primals_433, (48, ), (1, ))
    assert_size_stride(primals_434, (48, ), (1, ))
    assert_size_stride(primals_435, (48, ), (1, ))
    assert_size_stride(primals_436, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_437, (394, ), (1, ))
    assert_size_stride(primals_438, (394, ), (1, ))
    assert_size_stride(primals_439, (394, ), (1, ))
    assert_size_stride(primals_440, (394, ), (1, ))
    assert_size_stride(primals_441, (197, 394, 1, 1), (394, 1, 1, 1))
    assert_size_stride(primals_442, (197, ), (1, ))
    assert_size_stride(primals_443, (197, ), (1, ))
    assert_size_stride(primals_444, (197, ), (1, ))
    assert_size_stride(primals_445, (197, ), (1, ))
    assert_size_stride(primals_446, (48, 197, 1, 1), (197, 1, 1, 1))
    assert_size_stride(primals_447, (48, ), (1, ))
    assert_size_stride(primals_448, (48, ), (1, ))
    assert_size_stride(primals_449, (48, ), (1, ))
    assert_size_stride(primals_450, (48, ), (1, ))
    assert_size_stride(primals_451, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_452, (209, ), (1, ))
    assert_size_stride(primals_453, (209, ), (1, ))
    assert_size_stride(primals_454, (209, ), (1, ))
    assert_size_stride(primals_455, (209, ), (1, ))
    assert_size_stride(primals_456, (48, 209, 1, 1), (209, 1, 1, 1))
    assert_size_stride(primals_457, (48, ), (1, ))
    assert_size_stride(primals_458, (48, ), (1, ))
    assert_size_stride(primals_459, (48, ), (1, ))
    assert_size_stride(primals_460, (48, ), (1, ))
    assert_size_stride(primals_461, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_462, (221, ), (1, ))
    assert_size_stride(primals_463, (221, ), (1, ))
    assert_size_stride(primals_464, (221, ), (1, ))
    assert_size_stride(primals_465, (221, ), (1, ))
    assert_size_stride(primals_466, (48, 221, 1, 1), (221, 1, 1, 1))
    assert_size_stride(primals_467, (48, ), (1, ))
    assert_size_stride(primals_468, (48, ), (1, ))
    assert_size_stride(primals_469, (48, ), (1, ))
    assert_size_stride(primals_470, (48, ), (1, ))
    assert_size_stride(primals_471, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_472, (233, ), (1, ))
    assert_size_stride(primals_473, (233, ), (1, ))
    assert_size_stride(primals_474, (233, ), (1, ))
    assert_size_stride(primals_475, (233, ), (1, ))
    assert_size_stride(primals_476, (48, 233, 1, 1), (233, 1, 1, 1))
    assert_size_stride(primals_477, (48, ), (1, ))
    assert_size_stride(primals_478, (48, ), (1, ))
    assert_size_stride(primals_479, (48, ), (1, ))
    assert_size_stride(primals_480, (48, ), (1, ))
    assert_size_stride(primals_481, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_482, (245, ), (1, ))
    assert_size_stride(primals_483, (245, ), (1, ))
    assert_size_stride(primals_484, (245, ), (1, ))
    assert_size_stride(primals_485, (245, ), (1, ))
    assert_size_stride(primals_486, (48, 245, 1, 1), (245, 1, 1, 1))
    assert_size_stride(primals_487, (48, ), (1, ))
    assert_size_stride(primals_488, (48, ), (1, ))
    assert_size_stride(primals_489, (48, ), (1, ))
    assert_size_stride(primals_490, (48, ), (1, ))
    assert_size_stride(primals_491, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_492, (257, ), (1, ))
    assert_size_stride(primals_493, (257, ), (1, ))
    assert_size_stride(primals_494, (257, ), (1, ))
    assert_size_stride(primals_495, (257, ), (1, ))
    assert_size_stride(primals_496, (48, 257, 1, 1), (257, 1, 1, 1))
    assert_size_stride(primals_497, (48, ), (1, ))
    assert_size_stride(primals_498, (48, ), (1, ))
    assert_size_stride(primals_499, (48, ), (1, ))
    assert_size_stride(primals_500, (48, ), (1, ))
    assert_size_stride(primals_501, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_502, (269, ), (1, ))
    assert_size_stride(primals_503, (269, ), (1, ))
    assert_size_stride(primals_504, (269, ), (1, ))
    assert_size_stride(primals_505, (269, ), (1, ))
    assert_size_stride(primals_506, (48, 269, 1, 1), (269, 1, 1, 1))
    assert_size_stride(primals_507, (48, ), (1, ))
    assert_size_stride(primals_508, (48, ), (1, ))
    assert_size_stride(primals_509, (48, ), (1, ))
    assert_size_stride(primals_510, (48, ), (1, ))
    assert_size_stride(primals_511, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_512, (281, ), (1, ))
    assert_size_stride(primals_513, (281, ), (1, ))
    assert_size_stride(primals_514, (281, ), (1, ))
    assert_size_stride(primals_515, (281, ), (1, ))
    assert_size_stride(primals_516, (48, 281, 1, 1), (281, 1, 1, 1))
    assert_size_stride(primals_517, (48, ), (1, ))
    assert_size_stride(primals_518, (48, ), (1, ))
    assert_size_stride(primals_519, (48, ), (1, ))
    assert_size_stride(primals_520, (48, ), (1, ))
    assert_size_stride(primals_521, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_522, (293, ), (1, ))
    assert_size_stride(primals_523, (293, ), (1, ))
    assert_size_stride(primals_524, (293, ), (1, ))
    assert_size_stride(primals_525, (293, ), (1, ))
    assert_size_stride(primals_526, (48, 293, 1, 1), (293, 1, 1, 1))
    assert_size_stride(primals_527, (48, ), (1, ))
    assert_size_stride(primals_528, (48, ), (1, ))
    assert_size_stride(primals_529, (48, ), (1, ))
    assert_size_stride(primals_530, (48, ), (1, ))
    assert_size_stride(primals_531, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_532, (305, ), (1, ))
    assert_size_stride(primals_533, (305, ), (1, ))
    assert_size_stride(primals_534, (305, ), (1, ))
    assert_size_stride(primals_535, (305, ), (1, ))
    assert_size_stride(primals_536, (48, 305, 1, 1), (305, 1, 1, 1))
    assert_size_stride(primals_537, (48, ), (1, ))
    assert_size_stride(primals_538, (48, ), (1, ))
    assert_size_stride(primals_539, (48, ), (1, ))
    assert_size_stride(primals_540, (48, ), (1, ))
    assert_size_stride(primals_541, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_542, (317, ), (1, ))
    assert_size_stride(primals_543, (317, ), (1, ))
    assert_size_stride(primals_544, (317, ), (1, ))
    assert_size_stride(primals_545, (317, ), (1, ))
    assert_size_stride(primals_546, (48, 317, 1, 1), (317, 1, 1, 1))
    assert_size_stride(primals_547, (48, ), (1, ))
    assert_size_stride(primals_548, (48, ), (1, ))
    assert_size_stride(primals_549, (48, ), (1, ))
    assert_size_stride(primals_550, (48, ), (1, ))
    assert_size_stride(primals_551, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_552, (329, ), (1, ))
    assert_size_stride(primals_553, (329, ), (1, ))
    assert_size_stride(primals_554, (329, ), (1, ))
    assert_size_stride(primals_555, (329, ), (1, ))
    assert_size_stride(primals_556, (48, 329, 1, 1), (329, 1, 1, 1))
    assert_size_stride(primals_557, (48, ), (1, ))
    assert_size_stride(primals_558, (48, ), (1, ))
    assert_size_stride(primals_559, (48, ), (1, ))
    assert_size_stride(primals_560, (48, ), (1, ))
    assert_size_stride(primals_561, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_562, (341, ), (1, ))
    assert_size_stride(primals_563, (341, ), (1, ))
    assert_size_stride(primals_564, (341, ), (1, ))
    assert_size_stride(primals_565, (341, ), (1, ))
    assert_size_stride(primals_566, (48, 341, 1, 1), (341, 1, 1, 1))
    assert_size_stride(primals_567, (48, ), (1, ))
    assert_size_stride(primals_568, (48, ), (1, ))
    assert_size_stride(primals_569, (48, ), (1, ))
    assert_size_stride(primals_570, (48, ), (1, ))
    assert_size_stride(primals_571, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_572, (353, ), (1, ))
    assert_size_stride(primals_573, (353, ), (1, ))
    assert_size_stride(primals_574, (353, ), (1, ))
    assert_size_stride(primals_575, (353, ), (1, ))
    assert_size_stride(primals_576, (48, 353, 1, 1), (353, 1, 1, 1))
    assert_size_stride(primals_577, (48, ), (1, ))
    assert_size_stride(primals_578, (48, ), (1, ))
    assert_size_stride(primals_579, (48, ), (1, ))
    assert_size_stride(primals_580, (48, ), (1, ))
    assert_size_stride(primals_581, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_582, (365, ), (1, ))
    assert_size_stride(primals_583, (365, ), (1, ))
    assert_size_stride(primals_584, (365, ), (1, ))
    assert_size_stride(primals_585, (365, ), (1, ))
    assert_size_stride(primals_586, (48, 365, 1, 1), (365, 1, 1, 1))
    assert_size_stride(primals_587, (48, ), (1, ))
    assert_size_stride(primals_588, (48, ), (1, ))
    assert_size_stride(primals_589, (48, ), (1, ))
    assert_size_stride(primals_590, (48, ), (1, ))
    assert_size_stride(primals_591, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_592, (377, ), (1, ))
    assert_size_stride(primals_593, (377, ), (1, ))
    assert_size_stride(primals_594, (377, ), (1, ))
    assert_size_stride(primals_595, (377, ), (1, ))
    assert_size_stride(primals_596, (48, 377, 1, 1), (377, 1, 1, 1))
    assert_size_stride(primals_597, (48, ), (1, ))
    assert_size_stride(primals_598, (48, ), (1, ))
    assert_size_stride(primals_599, (48, ), (1, ))
    assert_size_stride(primals_600, (48, ), (1, ))
    assert_size_stride(primals_601, (12, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_602, (389, ), (1, ))
    assert_size_stride(primals_603, (389, ), (1, ))
    assert_size_stride(primals_604, (389, ), (1, ))
    assert_size_stride(primals_605, (389, ), (1, ))
    assert_size_stride(primals_606, (1000, 389), (389, 1))
    assert_size_stride(primals_607, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 262144, grid=grid(262144), stream=stream0)
        del primals_6
        buf9 = empty_strided_cuda((4, 76, 16, 16), (19456, 256, 16, 1), torch.float32)
        buf2 = reinterpret_tensor(buf9, (4, 64, 16, 16), (19456, 256, 16, 1), 0)  # alias
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.int8)
        buf4 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf305 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_native_batch_norm_backward_relu_1.run(buf1, primals_7, primals_8, primals_9, primals_10, buf2, buf3, buf4, buf305, 65536, grid=grid(65536), stream=stream0)
        del primals_10
        del primals_7
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf6 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, primals_12, primals_13, primals_14, primals_15, buf6, 49152, grid=grid(49152), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf8 = reinterpret_tensor(buf9, (4, 12, 16, 16), (19456, 256, 16, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf7, buf8, 12288, grid=grid(12288), stream=stream0)
        buf10 = empty_strided_cuda((4, 76, 16, 16), (19456, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf9, primals_17, primals_18, primals_19, primals_20, buf10, 77824, grid=grid(77824), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf12 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf11, primals_22, primals_23, primals_24, primals_25, buf12, 49152, grid=grid(49152), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf14 = empty_strided_cuda((4, 88, 16, 16), (22528, 256, 16, 1), torch.float32)
        buf15 = empty_strided_cuda((4, 88, 16, 16), (22528, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, input_19, input_20], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf9, buf13, primals_27, primals_28, primals_29, primals_30, buf14, buf15, 90112, grid=grid(90112), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf17 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf16, primals_32, primals_33, primals_34, primals_35, buf17, 49152, grid=grid(49152), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf19 = empty_strided_cuda((4, 100, 16, 16), (25600, 256, 16, 1), torch.float32)
        buf20 = empty_strided_cuda((4, 100, 16, 16), (25600, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_25, input_26, input_27], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6.run(buf14, buf18, primals_37, primals_38, primals_39, primals_40, buf19, buf20, 102400, grid=grid(102400), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf22 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf21, primals_42, primals_43, primals_44, primals_45, buf22, 49152, grid=grid(49152), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf24 = empty_strided_cuda((4, 112, 16, 16), (28672, 256, 16, 1), torch.float32)
        buf25 = empty_strided_cuda((4, 112, 16, 16), (28672, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32, input_33, input_34], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7.run(buf19, buf23, primals_47, primals_48, primals_49, primals_50, buf24, buf25, 114688, grid=grid(114688), stream=stream0)
        del primals_50
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf27 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf26, primals_52, primals_53, primals_54, primals_55, buf27, 49152, grid=grid(49152), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf29 = empty_strided_cuda((4, 124, 16, 16), (31744, 256, 16, 1), torch.float32)
        buf30 = empty_strided_cuda((4, 124, 16, 16), (31744, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, input_40, input_41], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8.run(buf24, buf28, primals_57, primals_58, primals_59, primals_60, buf29, buf30, 126976, grid=grid(126976), stream=stream0)
        del primals_60
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 48, 16, 16), (12288, 256, 16, 1))
        buf32 = empty_strided_cuda((4, 48, 16, 16), (12288, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf31, primals_62, primals_63, primals_64, primals_65, buf32, 49152, grid=grid(49152), stream=stream0)
        del primals_65
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 12, 16, 16), (3072, 256, 16, 1))
        buf34 = empty_strided_cuda((4, 136, 16, 16), (34816, 256, 16, 1), torch.float32)
        buf35 = empty_strided_cuda((4, 136, 16, 16), (34816, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_46, input_47, input_48], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf29, buf33, primals_67, primals_68, primals_69, primals_70, buf34, buf35, 139264, grid=grid(139264), stream=stream0)
        del primals_70
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_71, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 68, 16, 16), (17408, 256, 16, 1))
        buf37 = empty_strided_cuda((4, 68, 8, 8), (4352, 64, 8, 1), torch.float32)
        buf38 = empty_strided_cuda((4, 68, 8, 8), (4352, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, input_51, input_52], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_10.run(buf36, primals_72, primals_73, primals_74, primals_75, buf37, buf38, 17408, grid=grid(17408), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf40 = reinterpret_tensor(buf33, (4, 48, 8, 8), (3072, 64, 8, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf39, primals_77, primals_78, primals_79, primals_80, buf40, 12288, grid=grid(12288), stream=stream0)
        del primals_80
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 12, 8, 8), (768, 64, 8, 1))
        buf42 = empty_strided_cuda((4, 80, 8, 8), (5120, 64, 8, 1), torch.float32)
        buf43 = empty_strided_cuda((4, 80, 8, 8), (5120, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_57, input_58, input_59], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf37, buf41, primals_82, primals_83, primals_84, primals_85, buf42, buf43, 20480, grid=grid(20480), stream=stream0)
        del buf41
        del primals_85
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf45 = reinterpret_tensor(buf28, (4, 48, 8, 8), (3072, 64, 8, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_61, input_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf44, primals_87, primals_88, primals_89, primals_90, buf45, 12288, grid=grid(12288), stream=stream0)
        del primals_90
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 12, 8, 8), (768, 64, 8, 1))
        buf47 = empty_strided_cuda((4, 92, 8, 8), (5888, 64, 8, 1), torch.float32)
        buf48 = empty_strided_cuda((4, 92, 8, 8), (5888, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_64, input_65, input_66], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13.run(buf42, buf46, primals_92, primals_93, primals_94, primals_95, buf47, buf48, 23552, grid=grid(23552), stream=stream0)
        del buf46
        del primals_95
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf50 = reinterpret_tensor(buf23, (4, 48, 8, 8), (3072, 64, 8, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf49, primals_97, primals_98, primals_99, primals_100, buf50, 12288, grid=grid(12288), stream=stream0)
        del primals_100
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 12, 8, 8), (768, 64, 8, 1))
        buf52 = empty_strided_cuda((4, 104, 8, 8), (6656, 64, 8, 1), torch.float32)
        buf53 = empty_strided_cuda((4, 104, 8, 8), (6656, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, input_72, input_73], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_14.run(buf47, buf51, primals_102, primals_103, primals_104, primals_105, buf52, buf53, 26624, grid=grid(26624), stream=stream0)
        del buf51
        del primals_105
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf55 = reinterpret_tensor(buf18, (4, 48, 8, 8), (3072, 64, 8, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_75, input_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf54, primals_107, primals_108, primals_109, primals_110, buf55, 12288, grid=grid(12288), stream=stream0)
        del primals_110
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 12, 8, 8), (768, 64, 8, 1))
        buf57 = empty_strided_cuda((4, 116, 8, 8), (7424, 64, 8, 1), torch.float32)
        buf58 = empty_strided_cuda((4, 116, 8, 8), (7424, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_78, input_79, input_80], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15.run(buf52, buf56, primals_112, primals_113, primals_114, primals_115, buf57, buf58, 29696, grid=grid(29696), stream=stream0)
        del buf56
        del primals_115
        # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf60 = reinterpret_tensor(buf13, (4, 48, 8, 8), (3072, 64, 8, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_82, input_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf59, primals_117, primals_118, primals_119, primals_120, buf60, 12288, grid=grid(12288), stream=stream0)
        del primals_120
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 12, 8, 8), (768, 64, 8, 1))
        buf62 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        buf63 = empty_strided_cuda((4, 128, 8, 8), (8192, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_85, input_86, input_87], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf57, buf61, primals_122, primals_123, primals_124, primals_125, buf62, buf63, 32768, grid=grid(32768), stream=stream0)
        del buf61
        del primals_125
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf65 = reinterpret_tensor(buf7, (4, 48, 8, 8), (3072, 64, 8, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_89, input_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf64, primals_127, primals_128, primals_129, primals_130, buf65, 12288, grid=grid(12288), stream=stream0)
        del primals_130
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_131, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 12, 8, 8), (768, 64, 8, 1))
        buf67 = empty_strided_cuda((4, 140, 8, 8), (8960, 64, 8, 1), torch.float32)
        buf68 = empty_strided_cuda((4, 140, 8, 8), (8960, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_92, input_93, input_94], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17.run(buf62, buf66, primals_132, primals_133, primals_134, primals_135, buf67, buf68, 35840, grid=grid(35840), stream=stream0)
        del buf66
        del primals_135
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf70 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_96, input_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf69, primals_137, primals_138, primals_139, primals_140, buf70, 12288, grid=grid(12288), stream=stream0)
        del primals_140
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 12, 8, 8), (768, 64, 8, 1))
        buf72 = empty_strided_cuda((4, 152, 8, 8), (9728, 64, 8, 1), torch.float32)
        buf73 = empty_strided_cuda((4, 152, 8, 8), (9728, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_99, input_100, input_101], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18.run(buf67, buf71, primals_142, primals_143, primals_144, primals_145, buf72, buf73, 38912, grid=grid(38912), stream=stream0)
        del buf71
        del primals_145
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf75 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_103, input_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf74, primals_147, primals_148, primals_149, primals_150, buf75, 12288, grid=grid(12288), stream=stream0)
        del primals_150
        # Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 12, 8, 8), (768, 64, 8, 1))
        buf77 = empty_strided_cuda((4, 164, 8, 8), (10496, 64, 8, 1), torch.float32)
        buf78 = empty_strided_cuda((4, 164, 8, 8), (10496, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_106, input_107, input_108], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf72, buf76, primals_152, primals_153, primals_154, primals_155, buf77, buf78, 41984, grid=grid(41984), stream=stream0)
        del buf76
        del primals_155
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf80 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_110, input_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf79, primals_157, primals_158, primals_159, primals_160, buf80, 12288, grid=grid(12288), stream=stream0)
        del primals_160
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 12, 8, 8), (768, 64, 8, 1))
        buf82 = empty_strided_cuda((4, 176, 8, 8), (11264, 64, 8, 1), torch.float32)
        buf83 = empty_strided_cuda((4, 176, 8, 8), (11264, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_113, input_114, input_115], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_20.run(buf77, buf81, primals_162, primals_163, primals_164, primals_165, buf82, buf83, 45056, grid=grid(45056), stream=stream0)
        del buf81
        del primals_165
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf85 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_117, input_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf84, primals_167, primals_168, primals_169, primals_170, buf85, 12288, grid=grid(12288), stream=stream0)
        del primals_170
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 12, 8, 8), (768, 64, 8, 1))
        buf87 = empty_strided_cuda((4, 188, 8, 8), (12032, 64, 8, 1), torch.float32)
        buf88 = empty_strided_cuda((4, 188, 8, 8), (12032, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_120, input_121, input_122], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21.run(buf82, buf86, primals_172, primals_173, primals_174, primals_175, buf87, buf88, 48128, grid=grid(48128), stream=stream0)
        del buf86
        del primals_175
        # Topologically Sorted Source Nodes: [input_123], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf90 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_124, input_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf89, primals_177, primals_178, primals_179, primals_180, buf90, 12288, grid=grid(12288), stream=stream0)
        del primals_180
        # Topologically Sorted Source Nodes: [input_126], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 12, 8, 8), (768, 64, 8, 1))
        buf92 = empty_strided_cuda((4, 200, 8, 8), (12800, 64, 8, 1), torch.float32)
        buf93 = empty_strided_cuda((4, 200, 8, 8), (12800, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_127, input_128, input_129], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_22.run(buf87, buf91, primals_182, primals_183, primals_184, primals_185, buf92, buf93, 51200, grid=grid(51200), stream=stream0)
        del buf91
        del primals_185
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf95 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_131, input_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf94, primals_187, primals_188, primals_189, primals_190, buf95, 12288, grid=grid(12288), stream=stream0)
        del primals_190
        # Topologically Sorted Source Nodes: [input_133], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 12, 8, 8), (768, 64, 8, 1))
        buf97 = empty_strided_cuda((4, 212, 8, 8), (13568, 64, 8, 1), torch.float32)
        buf98 = empty_strided_cuda((4, 212, 8, 8), (13568, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_134, input_135, input_136], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23.run(buf92, buf96, primals_192, primals_193, primals_194, primals_195, buf97, buf98, 54272, grid=grid(54272), stream=stream0)
        del buf96
        del primals_195
        # Topologically Sorted Source Nodes: [input_137], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 106, 8, 8), (6784, 64, 8, 1))
        buf100 = empty_strided_cuda((4, 106, 8, 8), (6784, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_138, input_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf99, primals_197, primals_198, primals_199, primals_200, buf100, 27136, grid=grid(27136), stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [input_140], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf102 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_141, input_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf101, primals_202, primals_203, primals_204, primals_205, buf102, 12288, grid=grid(12288), stream=stream0)
        del primals_205
        # Topologically Sorted Source Nodes: [input_143], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_206, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 12, 8, 8), (768, 64, 8, 1))
        buf104 = empty_strided_cuda((4, 118, 8, 8), (7552, 64, 8, 1), torch.float32)
        buf105 = empty_strided_cuda((4, 118, 8, 8), (7552, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_144, input_145, input_146], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_25.run(buf99, buf103, primals_207, primals_208, primals_209, primals_210, buf104, buf105, 30208, grid=grid(30208), stream=stream0)
        del buf103
        del primals_210
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf107 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_148, input_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf106, primals_212, primals_213, primals_214, primals_215, buf107, 12288, grid=grid(12288), stream=stream0)
        del primals_215
        # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_216, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 12, 8, 8), (768, 64, 8, 1))
        buf109 = empty_strided_cuda((4, 130, 8, 8), (8320, 64, 8, 1), torch.float32)
        buf110 = empty_strided_cuda((4, 130, 8, 8), (8320, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_151, input_152, input_153], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_26.run(buf104, buf108, primals_217, primals_218, primals_219, primals_220, buf109, buf110, 33280, grid=grid(33280), stream=stream0)
        del buf108
        del primals_220
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf112 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_155, input_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf111, primals_222, primals_223, primals_224, primals_225, buf112, 12288, grid=grid(12288), stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [input_157], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_226, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 12, 8, 8), (768, 64, 8, 1))
        buf114 = empty_strided_cuda((4, 142, 8, 8), (9088, 64, 8, 1), torch.float32)
        buf115 = empty_strided_cuda((4, 142, 8, 8), (9088, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_158, input_159, input_160], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_27.run(buf109, buf113, primals_227, primals_228, primals_229, primals_230, buf114, buf115, 36352, grid=grid(36352), stream=stream0)
        del buf113
        del primals_230
        # Topologically Sorted Source Nodes: [input_161], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf117 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_162, input_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf116, primals_232, primals_233, primals_234, primals_235, buf117, 12288, grid=grid(12288), stream=stream0)
        del primals_235
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_236, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 12, 8, 8), (768, 64, 8, 1))
        buf119 = empty_strided_cuda((4, 154, 8, 8), (9856, 64, 8, 1), torch.float32)
        buf120 = empty_strided_cuda((4, 154, 8, 8), (9856, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_165, input_166, input_167], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_28.run(buf114, buf118, primals_237, primals_238, primals_239, primals_240, buf119, buf120, 39424, grid=grid(39424), stream=stream0)
        del buf118
        del primals_240
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf122 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_169, input_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf121, primals_242, primals_243, primals_244, primals_245, buf122, 12288, grid=grid(12288), stream=stream0)
        del primals_245
        # Topologically Sorted Source Nodes: [input_171], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_246, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 12, 8, 8), (768, 64, 8, 1))
        buf124 = empty_strided_cuda((4, 166, 8, 8), (10624, 64, 8, 1), torch.float32)
        buf125 = empty_strided_cuda((4, 166, 8, 8), (10624, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_172, input_173, input_174], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_29.run(buf119, buf123, primals_247, primals_248, primals_249, primals_250, buf124, buf125, 42496, grid=grid(42496), stream=stream0)
        del buf123
        del primals_250
        # Topologically Sorted Source Nodes: [input_175], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf127 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_176, input_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf126, primals_252, primals_253, primals_254, primals_255, buf127, 12288, grid=grid(12288), stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [input_178], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_256, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 12, 8, 8), (768, 64, 8, 1))
        buf129 = empty_strided_cuda((4, 178, 8, 8), (11392, 64, 8, 1), torch.float32)
        buf130 = empty_strided_cuda((4, 178, 8, 8), (11392, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_179, input_180, input_181], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_30.run(buf124, buf128, primals_257, primals_258, primals_259, primals_260, buf129, buf130, 45568, grid=grid(45568), stream=stream0)
        del buf128
        del primals_260
        # Topologically Sorted Source Nodes: [input_182], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf132 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_183, input_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf131, primals_262, primals_263, primals_264, primals_265, buf132, 12288, grid=grid(12288), stream=stream0)
        del primals_265
        # Topologically Sorted Source Nodes: [input_185], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_266, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 12, 8, 8), (768, 64, 8, 1))
        buf134 = empty_strided_cuda((4, 190, 8, 8), (12160, 64, 8, 1), torch.float32)
        buf135 = empty_strided_cuda((4, 190, 8, 8), (12160, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_186, input_187, input_188], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_31.run(buf129, buf133, primals_267, primals_268, primals_269, primals_270, buf134, buf135, 48640, grid=grid(48640), stream=stream0)
        del buf133
        del primals_270
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf137 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_190, input_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf136, primals_272, primals_273, primals_274, primals_275, buf137, 12288, grid=grid(12288), stream=stream0)
        del primals_275
        # Topologically Sorted Source Nodes: [input_192], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_276, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 12, 8, 8), (768, 64, 8, 1))
        buf139 = empty_strided_cuda((4, 202, 8, 8), (12928, 64, 8, 1), torch.float32)
        buf140 = empty_strided_cuda((4, 202, 8, 8), (12928, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_193, input_194, input_195], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_32.run(buf134, buf138, primals_277, primals_278, primals_279, primals_280, buf139, buf140, 51712, grid=grid(51712), stream=stream0)
        del buf138
        del primals_280
        # Topologically Sorted Source Nodes: [input_196], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf142 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_197, input_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf141, primals_282, primals_283, primals_284, primals_285, buf142, 12288, grid=grid(12288), stream=stream0)
        del primals_285
        # Topologically Sorted Source Nodes: [input_199], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_286, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 12, 8, 8), (768, 64, 8, 1))
        buf144 = empty_strided_cuda((4, 214, 8, 8), (13696, 64, 8, 1), torch.float32)
        buf145 = empty_strided_cuda((4, 214, 8, 8), (13696, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_200, input_201, input_202], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_33.run(buf139, buf143, primals_287, primals_288, primals_289, primals_290, buf144, buf145, 54784, grid=grid(54784), stream=stream0)
        del buf143
        del primals_290
        # Topologically Sorted Source Nodes: [input_203], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf147 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_204, input_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf146, primals_292, primals_293, primals_294, primals_295, buf147, 12288, grid=grid(12288), stream=stream0)
        del primals_295
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_296, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 12, 8, 8), (768, 64, 8, 1))
        buf149 = empty_strided_cuda((4, 226, 8, 8), (14464, 64, 8, 1), torch.float32)
        buf150 = empty_strided_cuda((4, 226, 8, 8), (14464, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_207, input_208, input_209], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_34.run(buf144, buf148, primals_297, primals_298, primals_299, primals_300, buf149, buf150, 57856, grid=grid(57856), stream=stream0)
        del buf148
        del primals_300
        # Topologically Sorted Source Nodes: [input_210], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf152 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_211, input_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf151, primals_302, primals_303, primals_304, primals_305, buf152, 12288, grid=grid(12288), stream=stream0)
        del primals_305
        # Topologically Sorted Source Nodes: [input_213], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_306, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 12, 8, 8), (768, 64, 8, 1))
        buf154 = empty_strided_cuda((4, 238, 8, 8), (15232, 64, 8, 1), torch.float32)
        buf155 = empty_strided_cuda((4, 238, 8, 8), (15232, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_214, input_215, input_216], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_35.run(buf149, buf153, primals_307, primals_308, primals_309, primals_310, buf154, buf155, 60928, grid=grid(60928), stream=stream0)
        del buf153
        del primals_310
        # Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf157 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_218, input_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf156, primals_312, primals_313, primals_314, primals_315, buf157, 12288, grid=grid(12288), stream=stream0)
        del primals_315
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_316, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 12, 8, 8), (768, 64, 8, 1))
        buf159 = empty_strided_cuda((4, 250, 8, 8), (16000, 64, 8, 1), torch.float32)
        buf160 = empty_strided_cuda((4, 250, 8, 8), (16000, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_221, input_222, input_223], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_36.run(buf154, buf158, primals_317, primals_318, primals_319, primals_320, buf159, buf160, 64000, grid=grid(64000), stream=stream0)
        del buf158
        del primals_320
        # Topologically Sorted Source Nodes: [input_224], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf162 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_225, input_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf161, primals_322, primals_323, primals_324, primals_325, buf162, 12288, grid=grid(12288), stream=stream0)
        del primals_325
        # Topologically Sorted Source Nodes: [input_227], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_326, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 12, 8, 8), (768, 64, 8, 1))
        buf164 = empty_strided_cuda((4, 262, 8, 8), (16768, 64, 8, 1), torch.float32)
        buf165 = empty_strided_cuda((4, 262, 8, 8), (16768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_228, input_229, input_230], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_37.run(buf159, buf163, primals_327, primals_328, primals_329, primals_330, buf164, buf165, 67072, grid=grid(67072), stream=stream0)
        del buf163
        del primals_330
        # Topologically Sorted Source Nodes: [input_231], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_331, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf167 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_232, input_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf166, primals_332, primals_333, primals_334, primals_335, buf167, 12288, grid=grid(12288), stream=stream0)
        del primals_335
        # Topologically Sorted Source Nodes: [input_234], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_336, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 12, 8, 8), (768, 64, 8, 1))
        buf169 = empty_strided_cuda((4, 274, 8, 8), (17536, 64, 8, 1), torch.float32)
        buf170 = empty_strided_cuda((4, 274, 8, 8), (17536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_235, input_236, input_237], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38.run(buf164, buf168, primals_337, primals_338, primals_339, primals_340, buf169, buf170, 70144, grid=grid(70144), stream=stream0)
        del buf168
        del primals_340
        # Topologically Sorted Source Nodes: [input_238], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf172 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_239, input_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf171, primals_342, primals_343, primals_344, primals_345, buf172, 12288, grid=grid(12288), stream=stream0)
        del primals_345
        # Topologically Sorted Source Nodes: [input_241], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_346, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 12, 8, 8), (768, 64, 8, 1))
        buf174 = empty_strided_cuda((4, 286, 8, 8), (18304, 64, 8, 1), torch.float32)
        buf175 = empty_strided_cuda((4, 286, 8, 8), (18304, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_242, input_243, input_244], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39.run(buf169, buf173, primals_347, primals_348, primals_349, primals_350, buf174, buf175, 73216, grid=grid(73216), stream=stream0)
        del buf173
        del primals_350
        # Topologically Sorted Source Nodes: [input_245], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_351, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf177 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_246, input_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf176, primals_352, primals_353, primals_354, primals_355, buf177, 12288, grid=grid(12288), stream=stream0)
        del primals_355
        # Topologically Sorted Source Nodes: [input_248], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_356, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 12, 8, 8), (768, 64, 8, 1))
        buf179 = empty_strided_cuda((4, 298, 8, 8), (19072, 64, 8, 1), torch.float32)
        buf180 = empty_strided_cuda((4, 298, 8, 8), (19072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_249, input_250, input_251], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_40.run(buf174, buf178, primals_357, primals_358, primals_359, primals_360, buf179, buf180, 76288, grid=grid(76288), stream=stream0)
        del buf178
        del primals_360
        # Topologically Sorted Source Nodes: [input_252], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_361, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf182 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_253, input_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf181, primals_362, primals_363, primals_364, primals_365, buf182, 12288, grid=grid(12288), stream=stream0)
        del primals_365
        # Topologically Sorted Source Nodes: [input_255], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_366, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 12, 8, 8), (768, 64, 8, 1))
        buf184 = empty_strided_cuda((4, 310, 8, 8), (19840, 64, 8, 1), torch.float32)
        buf185 = empty_strided_cuda((4, 310, 8, 8), (19840, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_256, input_257, input_258], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_41.run(buf179, buf183, primals_367, primals_368, primals_369, primals_370, buf184, buf185, 79360, grid=grid(79360), stream=stream0)
        del buf183
        del primals_370
        # Topologically Sorted Source Nodes: [input_259], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_371, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf187 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_260, input_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf186, primals_372, primals_373, primals_374, primals_375, buf187, 12288, grid=grid(12288), stream=stream0)
        del primals_375
        # Topologically Sorted Source Nodes: [input_262], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_376, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 12, 8, 8), (768, 64, 8, 1))
        buf189 = empty_strided_cuda((4, 322, 8, 8), (20608, 64, 8, 1), torch.float32)
        buf190 = empty_strided_cuda((4, 322, 8, 8), (20608, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_263, input_264, input_265], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_42.run(buf184, buf188, primals_377, primals_378, primals_379, primals_380, buf189, buf190, 82432, grid=grid(82432), stream=stream0)
        del buf188
        del primals_380
        # Topologically Sorted Source Nodes: [input_266], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_381, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf192 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_267, input_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf191, primals_382, primals_383, primals_384, primals_385, buf192, 12288, grid=grid(12288), stream=stream0)
        del primals_385
        # Topologically Sorted Source Nodes: [input_269], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_386, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 12, 8, 8), (768, 64, 8, 1))
        buf194 = empty_strided_cuda((4, 334, 8, 8), (21376, 64, 8, 1), torch.float32)
        buf195 = empty_strided_cuda((4, 334, 8, 8), (21376, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_270, input_271, input_272], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43.run(buf189, buf193, primals_387, primals_388, primals_389, primals_390, buf194, buf195, 85504, grid=grid(85504), stream=stream0)
        del buf193
        del primals_390
        # Topologically Sorted Source Nodes: [input_273], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_391, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf197 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_274, input_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf196, primals_392, primals_393, primals_394, primals_395, buf197, 12288, grid=grid(12288), stream=stream0)
        del primals_395
        # Topologically Sorted Source Nodes: [input_276], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_396, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 12, 8, 8), (768, 64, 8, 1))
        buf199 = empty_strided_cuda((4, 346, 8, 8), (22144, 64, 8, 1), torch.float32)
        buf200 = empty_strided_cuda((4, 346, 8, 8), (22144, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_277, input_278, input_279], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_44.run(buf194, buf198, primals_397, primals_398, primals_399, primals_400, buf199, buf200, 88576, grid=grid(88576), stream=stream0)
        del buf198
        del primals_400
        # Topologically Sorted Source Nodes: [input_280], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_401, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf202 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_281, input_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf201, primals_402, primals_403, primals_404, primals_405, buf202, 12288, grid=grid(12288), stream=stream0)
        del primals_405
        # Topologically Sorted Source Nodes: [input_283], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_406, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 12, 8, 8), (768, 64, 8, 1))
        buf204 = empty_strided_cuda((4, 358, 8, 8), (22912, 64, 8, 1), torch.float32)
        buf205 = empty_strided_cuda((4, 358, 8, 8), (22912, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_284, input_285, input_286], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_45.run(buf199, buf203, primals_407, primals_408, primals_409, primals_410, buf204, buf205, 91648, grid=grid(91648), stream=stream0)
        del buf203
        del primals_410
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_411, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf207 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_288, input_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf206, primals_412, primals_413, primals_414, primals_415, buf207, 12288, grid=grid(12288), stream=stream0)
        del primals_415
        # Topologically Sorted Source Nodes: [input_290], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_416, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 12, 8, 8), (768, 64, 8, 1))
        buf209 = empty_strided_cuda((4, 370, 8, 8), (23680, 64, 8, 1), torch.float32)
        buf210 = empty_strided_cuda((4, 370, 8, 8), (23680, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_291, input_292, input_293], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_46.run(buf204, buf208, primals_417, primals_418, primals_419, primals_420, buf209, buf210, 94720, grid=grid(94720), stream=stream0)
        del buf208
        del primals_420
        # Topologically Sorted Source Nodes: [input_294], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_421, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf212 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_295, input_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf211, primals_422, primals_423, primals_424, primals_425, buf212, 12288, grid=grid(12288), stream=stream0)
        del primals_425
        # Topologically Sorted Source Nodes: [input_297], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_426, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 12, 8, 8), (768, 64, 8, 1))
        buf214 = empty_strided_cuda((4, 382, 8, 8), (24448, 64, 8, 1), torch.float32)
        buf215 = empty_strided_cuda((4, 382, 8, 8), (24448, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_298, input_299, input_300], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47.run(buf209, buf213, primals_427, primals_428, primals_429, primals_430, buf214, buf215, 97792, grid=grid(97792), stream=stream0)
        del buf213
        del primals_430
        # Topologically Sorted Source Nodes: [input_301], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_431, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf217 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_302, input_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf216, primals_432, primals_433, primals_434, primals_435, buf217, 12288, grid=grid(12288), stream=stream0)
        del primals_435
        # Topologically Sorted Source Nodes: [input_304], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_436, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 12, 8, 8), (768, 64, 8, 1))
        buf219 = empty_strided_cuda((4, 394, 8, 8), (25216, 64, 8, 1), torch.float32)
        buf220 = empty_strided_cuda((4, 394, 8, 8), (25216, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_305, input_306, input_307], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48.run(buf214, buf218, primals_437, primals_438, primals_439, primals_440, buf219, buf220, 100864, grid=grid(100864), stream=stream0)
        del buf218
        del primals_440
        # Topologically Sorted Source Nodes: [input_308], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_441, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 197, 8, 8), (12608, 64, 8, 1))
        buf222 = empty_strided_cuda((4, 197, 8, 8), (12608, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_309, input_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf221, primals_442, primals_443, primals_444, primals_445, buf222, 50432, grid=grid(50432), stream=stream0)
        del primals_445
        # Topologically Sorted Source Nodes: [input_311], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_446, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf224 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_312, input_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf223, primals_447, primals_448, primals_449, primals_450, buf224, 12288, grid=grid(12288), stream=stream0)
        del primals_450
        # Topologically Sorted Source Nodes: [input_314], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_451, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 12, 8, 8), (768, 64, 8, 1))
        buf226 = empty_strided_cuda((4, 209, 8, 8), (13376, 64, 8, 1), torch.float32)
        buf227 = empty_strided_cuda((4, 209, 8, 8), (13376, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_315, input_316, input_317], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_50.run(buf221, buf225, primals_452, primals_453, primals_454, primals_455, buf226, buf227, 53504, grid=grid(53504), stream=stream0)
        del buf225
        del primals_455
        # Topologically Sorted Source Nodes: [input_318], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_456, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf229 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_319, input_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf228, primals_457, primals_458, primals_459, primals_460, buf229, 12288, grid=grid(12288), stream=stream0)
        del primals_460
        # Topologically Sorted Source Nodes: [input_321], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_461, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 12, 8, 8), (768, 64, 8, 1))
        buf231 = empty_strided_cuda((4, 221, 8, 8), (14144, 64, 8, 1), torch.float32)
        buf232 = empty_strided_cuda((4, 221, 8, 8), (14144, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_322, input_323, input_324], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_51.run(buf226, buf230, primals_462, primals_463, primals_464, primals_465, buf231, buf232, 56576, grid=grid(56576), stream=stream0)
        del buf230
        del primals_465
        # Topologically Sorted Source Nodes: [input_325], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_466, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf234 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_326, input_327], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf233, primals_467, primals_468, primals_469, primals_470, buf234, 12288, grid=grid(12288), stream=stream0)
        del primals_470
        # Topologically Sorted Source Nodes: [input_328], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_471, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 12, 8, 8), (768, 64, 8, 1))
        buf236 = empty_strided_cuda((4, 233, 8, 8), (14912, 64, 8, 1), torch.float32)
        buf237 = empty_strided_cuda((4, 233, 8, 8), (14912, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_329, input_330, input_331], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_52.run(buf231, buf235, primals_472, primals_473, primals_474, primals_475, buf236, buf237, 59648, grid=grid(59648), stream=stream0)
        del buf235
        del primals_475
        # Topologically Sorted Source Nodes: [input_332], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_476, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf239 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_333, input_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf238, primals_477, primals_478, primals_479, primals_480, buf239, 12288, grid=grid(12288), stream=stream0)
        del primals_480
        # Topologically Sorted Source Nodes: [input_335], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_481, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 12, 8, 8), (768, 64, 8, 1))
        buf241 = empty_strided_cuda((4, 245, 8, 8), (15680, 64, 8, 1), torch.float32)
        buf242 = empty_strided_cuda((4, 245, 8, 8), (15680, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_336, input_337, input_338], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_53.run(buf236, buf240, primals_482, primals_483, primals_484, primals_485, buf241, buf242, 62720, grid=grid(62720), stream=stream0)
        del buf240
        del primals_485
        # Topologically Sorted Source Nodes: [input_339], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_486, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf244 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_340, input_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf243, primals_487, primals_488, primals_489, primals_490, buf244, 12288, grid=grid(12288), stream=stream0)
        del primals_490
        # Topologically Sorted Source Nodes: [input_342], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_491, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 12, 8, 8), (768, 64, 8, 1))
        buf246 = empty_strided_cuda((4, 257, 8, 8), (16448, 64, 8, 1), torch.float32)
        buf247 = empty_strided_cuda((4, 257, 8, 8), (16448, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_343, input_344, input_345], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_54.run(buf241, buf245, primals_492, primals_493, primals_494, primals_495, buf246, buf247, 65792, grid=grid(65792), stream=stream0)
        del buf245
        del primals_495
        # Topologically Sorted Source Nodes: [input_346], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_496, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf249 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_347, input_348], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf248, primals_497, primals_498, primals_499, primals_500, buf249, 12288, grid=grid(12288), stream=stream0)
        del primals_500
        # Topologically Sorted Source Nodes: [input_349], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_501, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 12, 8, 8), (768, 64, 8, 1))
        buf251 = empty_strided_cuda((4, 269, 8, 8), (17216, 64, 8, 1), torch.float32)
        buf252 = empty_strided_cuda((4, 269, 8, 8), (17216, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_350, input_351, input_352], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_55.run(buf246, buf250, primals_502, primals_503, primals_504, primals_505, buf251, buf252, 68864, grid=grid(68864), stream=stream0)
        del buf250
        del primals_505
        # Topologically Sorted Source Nodes: [input_353], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_506, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf254 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_354, input_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf253, primals_507, primals_508, primals_509, primals_510, buf254, 12288, grid=grid(12288), stream=stream0)
        del primals_510
        # Topologically Sorted Source Nodes: [input_356], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_511, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 12, 8, 8), (768, 64, 8, 1))
        buf256 = empty_strided_cuda((4, 281, 8, 8), (17984, 64, 8, 1), torch.float32)
        buf257 = empty_strided_cuda((4, 281, 8, 8), (17984, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_357, input_358, input_359], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_56.run(buf251, buf255, primals_512, primals_513, primals_514, primals_515, buf256, buf257, 71936, grid=grid(71936), stream=stream0)
        del buf255
        del primals_515
        # Topologically Sorted Source Nodes: [input_360], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_516, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf259 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_361, input_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf258, primals_517, primals_518, primals_519, primals_520, buf259, 12288, grid=grid(12288), stream=stream0)
        del primals_520
        # Topologically Sorted Source Nodes: [input_363], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_521, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 12, 8, 8), (768, 64, 8, 1))
        buf261 = empty_strided_cuda((4, 293, 8, 8), (18752, 64, 8, 1), torch.float32)
        buf262 = empty_strided_cuda((4, 293, 8, 8), (18752, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_364, input_365, input_366], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57.run(buf256, buf260, primals_522, primals_523, primals_524, primals_525, buf261, buf262, 75008, grid=grid(75008), stream=stream0)
        del buf260
        del primals_525
        # Topologically Sorted Source Nodes: [input_367], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_526, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf264 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_368, input_369], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf263, primals_527, primals_528, primals_529, primals_530, buf264, 12288, grid=grid(12288), stream=stream0)
        del primals_530
        # Topologically Sorted Source Nodes: [input_370], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_531, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 12, 8, 8), (768, 64, 8, 1))
        buf266 = empty_strided_cuda((4, 305, 8, 8), (19520, 64, 8, 1), torch.float32)
        buf267 = empty_strided_cuda((4, 305, 8, 8), (19520, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_371, input_372, input_373], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_58.run(buf261, buf265, primals_532, primals_533, primals_534, primals_535, buf266, buf267, 78080, grid=grid(78080), stream=stream0)
        del buf265
        del primals_535
        # Topologically Sorted Source Nodes: [input_374], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_536, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf269 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_375, input_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf268, primals_537, primals_538, primals_539, primals_540, buf269, 12288, grid=grid(12288), stream=stream0)
        del primals_540
        # Topologically Sorted Source Nodes: [input_377], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_541, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 12, 8, 8), (768, 64, 8, 1))
        buf271 = empty_strided_cuda((4, 317, 8, 8), (20288, 64, 8, 1), torch.float32)
        buf272 = empty_strided_cuda((4, 317, 8, 8), (20288, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_378, input_379, input_380], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_59.run(buf266, buf270, primals_542, primals_543, primals_544, primals_545, buf271, buf272, 81152, grid=grid(81152), stream=stream0)
        del buf270
        del primals_545
        # Topologically Sorted Source Nodes: [input_381], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_546, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf274 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_382, input_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf273, primals_547, primals_548, primals_549, primals_550, buf274, 12288, grid=grid(12288), stream=stream0)
        del primals_550
        # Topologically Sorted Source Nodes: [input_384], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_551, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 12, 8, 8), (768, 64, 8, 1))
        buf276 = empty_strided_cuda((4, 329, 8, 8), (21056, 64, 8, 1), torch.float32)
        buf277 = empty_strided_cuda((4, 329, 8, 8), (21056, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_385, input_386, input_387], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_60.run(buf271, buf275, primals_552, primals_553, primals_554, primals_555, buf276, buf277, 84224, grid=grid(84224), stream=stream0)
        del buf275
        del primals_555
        # Topologically Sorted Source Nodes: [input_388], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_556, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf279 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_389, input_390], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf278, primals_557, primals_558, primals_559, primals_560, buf279, 12288, grid=grid(12288), stream=stream0)
        del primals_560
        # Topologically Sorted Source Nodes: [input_391], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_561, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 12, 8, 8), (768, 64, 8, 1))
        buf281 = empty_strided_cuda((4, 341, 8, 8), (21824, 64, 8, 1), torch.float32)
        buf282 = empty_strided_cuda((4, 341, 8, 8), (21824, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_392, input_393, input_394], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_61.run(buf276, buf280, primals_562, primals_563, primals_564, primals_565, buf281, buf282, 87296, grid=grid(87296), stream=stream0)
        del buf280
        del primals_565
        # Topologically Sorted Source Nodes: [input_395], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_566, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf284 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_396, input_397], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf283, primals_567, primals_568, primals_569, primals_570, buf284, 12288, grid=grid(12288), stream=stream0)
        del primals_570
        # Topologically Sorted Source Nodes: [input_398], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_571, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 12, 8, 8), (768, 64, 8, 1))
        buf286 = empty_strided_cuda((4, 353, 8, 8), (22592, 64, 8, 1), torch.float32)
        buf287 = empty_strided_cuda((4, 353, 8, 8), (22592, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_399, input_400, input_401], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_62.run(buf281, buf285, primals_572, primals_573, primals_574, primals_575, buf286, buf287, 90368, grid=grid(90368), stream=stream0)
        del buf285
        del primals_575
        # Topologically Sorted Source Nodes: [input_402], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_576, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf289 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_403, input_404], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf288, primals_577, primals_578, primals_579, primals_580, buf289, 12288, grid=grid(12288), stream=stream0)
        del primals_580
        # Topologically Sorted Source Nodes: [input_405], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, primals_581, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (4, 12, 8, 8), (768, 64, 8, 1))
        buf291 = empty_strided_cuda((4, 365, 8, 8), (23360, 64, 8, 1), torch.float32)
        buf292 = empty_strided_cuda((4, 365, 8, 8), (23360, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_406, input_407, input_408], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_63.run(buf286, buf290, primals_582, primals_583, primals_584, primals_585, buf291, buf292, 93440, grid=grid(93440), stream=stream0)
        del buf290
        del primals_585
        # Topologically Sorted Source Nodes: [input_409], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_586, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf294 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_410, input_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf293, primals_587, primals_588, primals_589, primals_590, buf294, 12288, grid=grid(12288), stream=stream0)
        del primals_590
        # Topologically Sorted Source Nodes: [input_412], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, primals_591, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 12, 8, 8), (768, 64, 8, 1))
        buf296 = empty_strided_cuda((4, 377, 8, 8), (24128, 64, 8, 1), torch.float32)
        buf297 = empty_strided_cuda((4, 377, 8, 8), (24128, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_413, input_414, input_415], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_64.run(buf291, buf295, primals_592, primals_593, primals_594, primals_595, buf296, buf297, 96512, grid=grid(96512), stream=stream0)
        del buf295
        del primals_595
        # Topologically Sorted Source Nodes: [input_416], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_596, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 48, 8, 8), (3072, 64, 8, 1))
        buf299 = empty_strided_cuda((4, 48, 8, 8), (3072, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_417, input_418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf298, primals_597, primals_598, primals_599, primals_600, buf299, 12288, grid=grid(12288), stream=stream0)
        del primals_600
        # Topologically Sorted Source Nodes: [input_419], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_601, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 12, 8, 8), (768, 64, 8, 1))
        buf301 = empty_strided_cuda((4, 389, 8, 8), (24896, 64, 8, 1), torch.float32)
        buf302 = empty_strided_cuda((4, 389, 1, 1), (389, 1, 1556, 1556), torch.float32)
        buf303 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [input_420, input_421, out, adaptive_avg_pool2d], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_relu_65.run(buf303, buf296, buf300, primals_602, primals_603, primals_604, primals_605, buf301, 1556, 64, grid=grid(1556), stream=stream0)
        del buf300
        buf304 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_607, reinterpret_tensor(buf303, (4, 389), (389, 1), 0), reinterpret_tensor(primals_606, (389, 1000), (1, 389), 0), alpha=1, beta=1, out=buf304)
        del primals_607
    return (buf304, primals_1, primals_2, primals_3, primals_4, primals_5, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_49, primals_51, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, primals_64, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_86, primals_87, primals_88, primals_89, primals_91, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_104, primals_106, primals_107, primals_108, primals_109, primals_111, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_131, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_167, primals_168, primals_169, primals_171, primals_172, primals_173, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_182, primals_183, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_202, primals_203, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_212, primals_213, primals_214, primals_216, primals_217, primals_218, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_246, primals_247, primals_248, primals_249, primals_251, primals_252, primals_253, primals_254, primals_256, primals_257, primals_258, primals_259, primals_261, primals_262, primals_263, primals_264, primals_266, primals_267, primals_268, primals_269, primals_271, primals_272, primals_273, primals_274, primals_276, primals_277, primals_278, primals_279, primals_281, primals_282, primals_283, primals_284, primals_286, primals_287, primals_288, primals_289, primals_291, primals_292, primals_293, primals_294, primals_296, primals_297, primals_298, primals_299, primals_301, primals_302, primals_303, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_326, primals_327, primals_328, primals_329, primals_331, primals_332, primals_333, primals_334, primals_336, primals_337, primals_338, primals_339, primals_341, primals_342, primals_343, primals_344, primals_346, primals_347, primals_348, primals_349, primals_351, primals_352, primals_353, primals_354, primals_356, primals_357, primals_358, primals_359, primals_361, primals_362, primals_363, primals_364, primals_366, primals_367, primals_368, primals_369, primals_371, primals_372, primals_373, primals_374, primals_376, primals_377, primals_378, primals_379, primals_381, primals_382, primals_383, primals_384, primals_386, primals_387, primals_388, primals_389, primals_391, primals_392, primals_393, primals_394, primals_396, primals_397, primals_398, primals_399, primals_401, primals_402, primals_403, primals_404, primals_406, primals_407, primals_408, primals_409, primals_411, primals_412, primals_413, primals_414, primals_416, primals_417, primals_418, primals_419, primals_421, primals_422, primals_423, primals_424, primals_426, primals_427, primals_428, primals_429, primals_431, primals_432, primals_433, primals_434, primals_436, primals_437, primals_438, primals_439, primals_441, primals_442, primals_443, primals_444, primals_446, primals_447, primals_448, primals_449, primals_451, primals_452, primals_453, primals_454, primals_456, primals_457, primals_458, primals_459, primals_461, primals_462, primals_463, primals_464, primals_466, primals_467, primals_468, primals_469, primals_471, primals_472, primals_473, primals_474, primals_476, primals_477, primals_478, primals_479, primals_481, primals_482, primals_483, primals_484, primals_486, primals_487, primals_488, primals_489, primals_491, primals_492, primals_493, primals_494, primals_496, primals_497, primals_498, primals_499, primals_501, primals_502, primals_503, primals_504, primals_506, primals_507, primals_508, primals_509, primals_511, primals_512, primals_513, primals_514, primals_516, primals_517, primals_518, primals_519, primals_521, primals_522, primals_523, primals_524, primals_526, primals_527, primals_528, primals_529, primals_531, primals_532, primals_533, primals_534, primals_536, primals_537, primals_538, primals_539, primals_541, primals_542, primals_543, primals_544, primals_546, primals_547, primals_548, primals_549, primals_551, primals_552, primals_553, primals_554, primals_556, primals_557, primals_558, primals_559, primals_561, primals_562, primals_563, primals_564, primals_566, primals_567, primals_568, primals_569, primals_571, primals_572, primals_573, primals_574, primals_576, primals_577, primals_578, primals_579, primals_581, primals_582, primals_583, primals_584, primals_586, primals_587, primals_588, primals_589, primals_591, primals_592, primals_593, primals_594, primals_596, primals_597, primals_598, primals_599, primals_601, primals_602, primals_603, primals_604, primals_605, buf0, buf1, buf3, buf4, buf5, buf6, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf17, buf19, buf20, buf21, buf22, buf24, buf25, buf26, buf27, buf29, buf30, buf31, buf32, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf42, buf43, buf44, buf45, buf47, buf48, buf49, buf50, buf52, buf53, buf54, buf55, buf57, buf58, buf59, buf60, buf62, buf63, buf64, buf65, buf67, buf68, buf69, buf70, buf72, buf73, buf74, buf75, buf77, buf78, buf79, buf80, buf82, buf83, buf84, buf85, buf87, buf88, buf89, buf90, buf92, buf93, buf94, buf95, buf97, buf98, buf99, buf100, buf101, buf102, buf104, buf105, buf106, buf107, buf109, buf110, buf111, buf112, buf114, buf115, buf116, buf117, buf119, buf120, buf121, buf122, buf124, buf125, buf126, buf127, buf129, buf130, buf131, buf132, buf134, buf135, buf136, buf137, buf139, buf140, buf141, buf142, buf144, buf145, buf146, buf147, buf149, buf150, buf151, buf152, buf154, buf155, buf156, buf157, buf159, buf160, buf161, buf162, buf164, buf165, buf166, buf167, buf169, buf170, buf171, buf172, buf174, buf175, buf176, buf177, buf179, buf180, buf181, buf182, buf184, buf185, buf186, buf187, buf189, buf190, buf191, buf192, buf194, buf195, buf196, buf197, buf199, buf200, buf201, buf202, buf204, buf205, buf206, buf207, buf209, buf210, buf211, buf212, buf214, buf215, buf216, buf217, buf219, buf220, buf221, buf222, buf223, buf224, buf226, buf227, buf228, buf229, buf231, buf232, buf233, buf234, buf236, buf237, buf238, buf239, buf241, buf242, buf243, buf244, buf246, buf247, buf248, buf249, buf251, buf252, buf253, buf254, buf256, buf257, buf258, buf259, buf261, buf262, buf263, buf264, buf266, buf267, buf268, buf269, buf271, buf272, buf273, buf274, buf276, buf277, buf278, buf279, buf281, buf282, buf283, buf284, buf286, buf287, buf288, buf289, buf291, buf292, buf293, buf294, buf296, buf297, buf298, buf299, buf301, reinterpret_tensor(buf303, (4, 389), (389, 1), 0), primals_606, buf305, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((48, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((76, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((76, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((76, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((76, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((48, 76, 1, 1), (76, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((48, 88, 1, 1), (88, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((48, 100, 1, 1), (100, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((48, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((48, 124, 1, 1), (124, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((136, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((136, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((136, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((136, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((68, 136, 1, 1), (136, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((68, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((48, 68, 1, 1), (68, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((48, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((48, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((48, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((48, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((48, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((48, 140, 1, 1), (140, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((48, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((164, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((164, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((164, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((164, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((48, 164, 1, 1), (164, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((48, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((188, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((188, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((188, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((188, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((48, 188, 1, 1), (188, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((48, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((212, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((212, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((212, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((212, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((106, 212, 1, 1), (212, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((48, 106, 1, 1), (106, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((118, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((118, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((118, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((118, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((48, 118, 1, 1), (118, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((130, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((130, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((130, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((130, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((48, 130, 1, 1), (130, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((142, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((142, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((142, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((142, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((48, 142, 1, 1), (142, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((154, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((154, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((154, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((154, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((48, 154, 1, 1), (154, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((166, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((166, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((166, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((166, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((48, 166, 1, 1), (166, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((178, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((178, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((178, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((178, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((48, 178, 1, 1), (178, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((190, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((190, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((190, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((190, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((48, 190, 1, 1), (190, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((202, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((202, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((202, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((202, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((48, 202, 1, 1), (202, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((214, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((214, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((214, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((214, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((48, 214, 1, 1), (214, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((226, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((226, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((226, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((226, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((48, 226, 1, 1), (226, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((238, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((238, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((238, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((238, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((48, 238, 1, 1), (238, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((250, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((250, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((250, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((250, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((48, 250, 1, 1), (250, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((262, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((262, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((262, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((262, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((48, 262, 1, 1), (262, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((274, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((274, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((274, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((274, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((48, 274, 1, 1), (274, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((286, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((286, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((286, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((286, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((48, 286, 1, 1), (286, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((298, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((298, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((298, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((298, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((48, 298, 1, 1), (298, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((310, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((310, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((310, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((310, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((48, 310, 1, 1), (310, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((322, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((322, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((322, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((322, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((48, 322, 1, 1), (322, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((334, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((334, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((334, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((334, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((48, 334, 1, 1), (334, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((346, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((346, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((346, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((346, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((48, 346, 1, 1), (346, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((358, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((358, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((358, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((358, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((48, 358, 1, 1), (358, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((370, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((370, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((370, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((370, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((48, 370, 1, 1), (370, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((382, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((382, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((382, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((382, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((48, 382, 1, 1), (382, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((394, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((394, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((394, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((394, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((197, 394, 1, 1), (394, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((197, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((197, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((197, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((197, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((48, 197, 1, 1), (197, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((209, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((209, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((209, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((209, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((48, 209, 1, 1), (209, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((221, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((221, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((221, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((221, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((48, 221, 1, 1), (221, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((233, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((233, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((233, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((233, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((48, 233, 1, 1), (233, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((245, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((245, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((245, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((245, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((48, 245, 1, 1), (245, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((257, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((257, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((257, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((257, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((48, 257, 1, 1), (257, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((269, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((269, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((269, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((269, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((48, 269, 1, 1), (269, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((281, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((281, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((281, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((281, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((48, 281, 1, 1), (281, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((293, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((293, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((293, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((293, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((48, 293, 1, 1), (293, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((305, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((305, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((305, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((305, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((48, 305, 1, 1), (305, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((317, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((317, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((317, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((317, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((48, 317, 1, 1), (317, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((329, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((329, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((329, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((329, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((48, 329, 1, 1), (329, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((341, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((341, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((341, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((341, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((48, 341, 1, 1), (341, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((353, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((353, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((353, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((353, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((48, 353, 1, 1), (353, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((365, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((365, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((365, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((365, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((48, 365, 1, 1), (365, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((377, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((377, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((377, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((377, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((48, 377, 1, 1), (377, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((12, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((389, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((389, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((389, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((389, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((1000, 389), (389, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
